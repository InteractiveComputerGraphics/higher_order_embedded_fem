use std::error::Error;
use std::ffi::OsStr;
use std::fs;
use std::process::Command;

use ignore::WalkBuilder;

fn git_command_stdout<I: IntoIterator<Item = S>, S: AsRef<OsStr>>(args: I) -> Result<String, Box<dyn Error>> {
    Ok(
        String::from_utf8_lossy(Command::new("git").args(args).output()?.stdout.as_slice())
            .trim_end_matches("\n")
            .replace("\n", ";"),
    )
}

fn main() -> Result<(), Box<dyn Error>> {
    // Prepare environment variables that will be baked into the binary
    {
        let last_commit = git_command_stdout(&["show", "-s", "--format=Commit: %H%nAuthor: %an, %aI%nTitle: '%s'"])?;

        let current_changes = git_command_stdout(&["status", "-b", "--porcelain"])?;

        let hostname = hostname::get()?.to_string_lossy().into_owned();
        let timestamp = chrono::Local::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, false);

        println!("cargo:rustc-env=FEMPROTO2_BUILD_TIMESTAMP={}", timestamp);
        println!("cargo:rustc-env=FEMPROTO2_BUILD_HOSTNAME={}", hostname);
        println!("cargo:rustc-env=FEMPROTO2_GIT_LAST_COMMIT={}", last_commit);
        println!("cargo:rustc-env=FEMPROTO2_GIT_CHANGES={}", current_changes);
    }

    // List directories that should be watched by cargo for changes for a rebuild.
    // Using the ignore crate allows to skip folders mentioned in .gitignore files,
    // e.g. changes in the target or data output directory should not lead to a rebuild.
    //
    // The .git folder has to be treated separately as every git command (even read only commands)
    // `touch` the .git folder. So we only want to rebuild on changes inside of the .git folder
    // (to cause a rebuild on a local commit).
    {
        // Add all top-level files and folders in the ../.git directory to watch
        for entry in fs::read_dir("../.git")?
            // Skip access errors
            .filter_map(|e| e.ok())
            // Skip symlinks
            .filter(|e| e.metadata().is_ok())
        {
            println!("cargo:rerun-if-changed={}", entry.path().to_str().unwrap());
        }

        // Add all other folders recursively from ../ that are not excluded per .gitignore etc.
        for entry in WalkBuilder::new("../")
            .add_custom_ignore_filename(".git")
            .follow_links(false)
            .build()
            .filter_map(|e| e.ok())
        {
            if entry.path().is_dir() {
                println!("cargo:rerun-if-changed={}", entry.path().to_str().unwrap());
            }
        }
    }

    Ok(())
}
