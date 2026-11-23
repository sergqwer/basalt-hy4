#include "serve.hpp"

#include "basalt/log.hpp"
#include "basalt/platform/file.hpp"
#include "basalt/platform/process.hpp"
#include "basalt/platform/system.hpp"

#include <cerrno>
#include <cstdlib>
#include <cstring>

namespace basalt::program::cli {

namespace {

#ifdef _WIN32
constexpr char kSep = '\\';
#else
constexpr char kSep = '/';
#endif

// A path made absolute even when the file is not there yet (a model the launcher will fetch): its folder's real path
// and its name. Empty when neither resolves.
std::string absolute_or_parent(const std::string& path) {
    std::string real;
    if (basalt::platform::absolute_path(path, real)) return real;
    const size_t cut = path.find_last_of("/\\");
    const std::string dir = cut == std::string::npos ? "." : (cut == 0 ? path.substr(0, 1) : path.substr(0, cut));
    const std::string name = cut == std::string::npos ? path : path.substr(cut + 1);
    if (name.empty() || !basalt::platform::absolute_path(dir, real)) return "";
    return real + (real.back() == '/' || real.back() == '\\' ? "" : std::string(1, kSep)) + name;
}

bool is_file(const std::string& path) {
    uint64_t size = 0;
    return basalt::platform::file_size(path, size);
}

// tools/get_model.py in the source tree, empty when it is not there
std::string get_model_tool(const std::string& src_dir) {
    const std::string tool = src_dir + kSep + "tools" + kSep + "get_model.py";
    return is_file(tool) ? tool : "";
}

}  // namespace

/// `basalt download [OWNER/REPO/FILE | PACK-NAME] [--dir DIR]`: fetches a model pack from Hugging Face ahead of a
/// start (tools/get_model.py download; the default is the Hy4-flash GPTQ pack). Returns the tool's exit code.
int download_subcommand(int argc, char** argv) {
    const std::string src_dir = basalt::platform::install_root();
    const std::string tool = src_dir.empty() ? "" : get_model_tool(src_dir);
    if (tool.empty()) { basalt::log::error("basalt download", "tools/get_model.py not found in the source tree"); return 1; }
    std::vector<std::string> args{basalt::platform::python_interpreter(src_dir), tool, "download"};
    for (int i = 2; i < argc; ++i) args.push_back(argv[i]);
    const int rc = basalt::platform::spawn_and_wait(args);
    if (rc < 0) basalt::log::error("basalt download", "cannot run %s: %s", args[0].c_str(), std::strerror(errno));
    return rc < 0 ? 1 : rc;
}

/// `basalt serve --config FILE` or `basalt serve MODEL.basalt [--python PY] [--server python|rust] [server args...]`:
/// runs the server over this binary: the Python one (serve/, the default) or its Rust port (serve-rs: basalt-serve
/// beside this binary or in serve-rs/target/{release,debug}), with the same arguments.  The server gets `--engine-bin` and what was typed; a MODEL.basalt argument stands for `--model MODEL.basalt`.
/// A model file that is not there is fetched first (tools/get_model.py ensure: the config entry's "hf", or a known pack
/// name), unless `--no-download` is given or the entry says "download": false.
/// Only returns on failure.
int serve_subcommand(int argc, char** argv) {
    std::string python;
    std::string server = "python";
    std::vector<std::string> extra;
    bool have_model = false, no_download = false;
    std::string model_path, config_path;
    // the working directory changes below: the paths the server reads must not depend on it
    auto absolute = [](const std::string& path) {
        std::string real;
        if (!basalt::platform::absolute_path(path, real)) {
            basalt::log::error("basalt serve", "cannot open %s", path.c_str());
            std::exit(1);
        }
        return real;
    };
    auto model_absolute = [](const std::string& path) {   // the model may be missing: it is fetched below
        const std::string real = absolute_or_parent(path);
        if (real.empty()) {
            basalt::log::error("basalt serve", "cannot open %s", path.c_str());
            std::exit(1);
        }
        return real;
    };
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        auto value = [&]() -> std::string {
            if (i + 1 >= argc) { basalt::log::error("basalt serve", "%s needs a value", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        if (a == "--python") python = value();
        else if (a == "--server") {
            server = value();
            if (server != "python" && server != "rust") {
                basalt::log::error("basalt serve", "--server: python or rust, not %s", server.c_str());
                std::exit(2);
            }
        }
        else if (a == "--no-download") no_download = true;
        else if (a == "--model") { model_path = model_absolute(value()); extra.insert(extra.end(), {a, model_path}); have_model = true; }
        else if (a == "--config") { config_path = absolute(value()); extra.insert(extra.end(), {a, config_path}); }
        // only the first argument can be MODEL.basalt; a later bare word is the value of the option before it
        else if (i == 2 && !have_model && !a.empty() && a[0] != '-') {
            model_path = model_absolute(a);
            extra.insert(extra.end(), {"--model", model_path});
            have_model = true;
        }
        else extra.push_back(a);
    }
    if (extra.empty()) {
        basalt::log::error("basalt", "usage: basalt serve --config FILE | MODEL.basalt [--python PY] [--server python|rust] [server args...]");
        return 2;
    }
    const std::string exe = basalt::platform::module_path();
    if (exe.empty()) { basalt::log::error("basalt serve", "cannot find this binary"); return 1; }
    const std::string src_dir = basalt::platform::install_root();
    if (src_dir.empty()) { basalt::log::error("basalt serve", "cannot find the directory above this binary"); return 1; }
    std::string program;
    if (server == "rust") {
#ifdef _WIN32
        const char* name = "basalt-serve.exe";
        const char sep = '\\';
#else
        const char* name = "basalt-serve";
        const char sep = '/';
#endif
        const std::string exe_dir = exe.substr(0, exe.find_last_of("/\\"));
        for (const std::string& dir : {exe_dir, src_dir + sep + "serve-rs" + sep + "target" + sep + "release",
                                       src_dir + sep + "serve-rs" + sep + "target" + sep + "debug"}) {
            const std::string candidate = dir + sep + name;
            uint64_t size = 0;
            if (basalt::platform::file_size(candidate, size)) { program = candidate; break; }
        }
        if (program.empty()) {
            basalt::log::error("basalt serve", "--server rust: %s not found beside %s or in %s%cserve-rs%ctarget", name,
                               exe.c_str(), src_dir.c_str(), sep, sep);
            return 1;
        }
    } else if (python.empty()) {
        python = basalt::platform::python_interpreter(src_dir);
    }
    // a missing model is fetched before the server starts; a config's entries are read by the tool, as the server reads them
    if (!no_download && ((have_model && !is_file(model_path)) || !config_path.empty())) {
        const std::string tool = get_model_tool(src_dir);
        const std::string py = python.empty() ? basalt::platform::python_interpreter(src_dir) : python;
        std::vector<std::string> fetch{py, tool, "ensure"};
        if (have_model) fetch.insert(fetch.end(), {"--model", model_path});
        else fetch.insert(fetch.end(), {"--config", config_path});
        const int rc = tool.empty() ? -1 : basalt::platform::spawn_and_wait(fetch);
        if (rc > 0) return rc;   // the tool said why
        if (rc < 0 && have_model) {
            basalt::log::error("basalt serve", "%s is missing and cannot be fetched (%s)", model_path.c_str(),
                               tool.empty() ? "no tools/get_model.py" : "no Python to run tools/get_model.py");
            return 1;
        }
        if (rc < 0) basalt::log::warn("basalt serve", "cannot check the config's model files (no Python for tools/get_model.py)");
    }
    if (basalt::platform::set_working_dir(src_dir) != 0) {
        basalt::log::error("basalt serve", "cannot enter %s", src_dir.c_str());
        return 1;
    }
    std::vector<std::string> args = server == "rust" ? std::vector<std::string>{program, "--engine-bin", exe}
                                                     : std::vector<std::string>{python, "-m", "serve.server", "--engine-bin", exe};
    args.insert(args.end(), extra.begin(), extra.end());
    // Does not return when it works. Windows has no exec, so this process waits for the server and ends with the
    // server's exit code. Reaching the line below means the server never started, and errno says why.
    basalt::platform::exec_replacing(args);
    basalt::log::error("basalt serve", "cannot run %s: %s", args[0].c_str(), std::strerror(errno));
    return 1;
}

}  // namespace basalt::program::cli
