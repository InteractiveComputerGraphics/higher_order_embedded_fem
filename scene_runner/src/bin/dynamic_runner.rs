use std::cmp::min;
use std::env;
use std::error::Error;
use std::fmt;
use std::fmt::Display;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use coarse_prof::profile;
use global_stash::stash_scope;
use log::{error, info, warn};
use structopt::StructOpt;

use hamilton::{register_component, Component, FilterSystem, StorageContainer, System, SystemCollection, Systems};
use simulation_toolbox::components::{
    get_export_sequence_index, get_gravity, get_step_index, get_time_step, set_gravity, ExportSequenceIndex, Gravity,
    SimulationTime, StepIndex, TimeStep, TimingSystem,
};
use simulation_toolbox::io::json_helper;
use simulation_toolbox::io::ply::{
    PlyFem2dOutput, PlyInterpolatedPoints2dOutput, PlyPolyMesh2dOutput, PlyPolyMesh3dOutput, PlyVolumeMesh2dOutput,
};
use simulation_toolbox::io::vtk::{VtkFemOutput, VtkSurfaceMeshOutput, VtkVolumeMeshOutput};

use scene_runner::scenes::{available_scenes, load_scene, Scene, SceneParameters};

static BUILD_TIMESTAMP: Option<&'static str> = option_env!("FEMPROTO2_BUILD_TIMESTAMP");
static BUILD_HOSTNAME: Option<&'static str> = option_env!("FEMPROTO2_BUILD_HOSTNAME");
static GIT_LAST_COMMIT: Option<&'static str> = option_env!("FEMPROTO2_GIT_LAST_COMMIT");
static GIT_CHANGES: Option<&'static str> = option_env!("FEMPROTO2_GIT_CHANGES");

#[derive(Debug, StructOpt)]
struct CommandlineArgs {
    #[structopt(
        short = "-s",
        long = "--scene",
        help = "Name of scene to simulate",
        required_unless = "list-scenes"
    )]
    scene: Option<String>,
    #[structopt(short = "-l", long = "--list-scenes", help = "List available scenes")]
    list_scenes: bool,
    #[structopt(long, help = "Enable PLY mesh data output")]
    output_ply: bool,
    #[structopt(long, help = "Disable VTK mesh data output")]
    no_output_vtk: bool,
    #[structopt(
        long,
        default_value = "data",
        parse(from_os_str),
        help = "Base directory for output files"
    )]
    output_dir: PathBuf,
    #[structopt(
        long,
        default_value = "assets",
        parse(from_os_str),
        help = "Base directory for asset files"
    )]
    asset_dir: PathBuf,
    #[structopt(long = "--output-fps", help = "Override output systems to output at a fixed fps")]
    output_fps: Option<f64>,
    #[structopt(long, help = "Overrides duration of scene")]
    duration: Option<f64>,
    #[structopt(long = "--dt", help = "Overrides time step")]
    timestep: Option<f64>,
    #[structopt(
        long,
        help = "Enables printing solver timings at every file export (with export FPS)"
    )]
    print_timings_output: bool,
    #[structopt(long, parse(from_os_str), help = "Configuration file for the scene")]
    config_file: Option<PathBuf>,
    #[structopt(
        long,
        parse(from_os_str),
        help = "Path for the logfile relative to 'output-dir/scene-name'"
    )]
    log_file: Option<PathBuf>,
    #[structopt(
        long,
        default_value = "0",
        help = "The number of threads to use for the rayon thread pool, if not specified it will be read from env or default rayon value"
    )]
    num_threads: usize,
    #[structopt(
        long,
        help = "Enables writing out the global statistics stash to file on every timestep"
    )]
    dump_stash: bool,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = CommandlineArgs::from_args();
    if args.list_scenes {
        list_scenes()?;
    } else {
        initialize_logging(&args)?;
        initialize_thread_pool(&args)?;

        info!("Started dynamic_runner");
        print_git_info()?;

        info!("Running on'{}'", hostname::get()?.to_string_lossy());
        info!("Executable path: '{}'", env::current_exe()?.to_string_lossy());
        info!("Working directory: '{}'", env::current_dir()?.to_string_lossy());
        info!("Full command line: '{}'", env::args().collect::<Vec<_>>().join(" "));

        if let Err(err) = run_scene(&args) {
            error!("Scene returned error: {}", err);
            error!("Aborting.");
        } else {
            info!("Exiting.");
        }
    }

    Ok(())
}

fn list_scenes() -> Result<(), Box<dyn Error>> {
    println!("Available scenes: ");
    for scene in available_scenes() {
        println!("  - {}", scene);
    }
    Ok(())
}

fn run_scene(args: &CommandlineArgs) -> Result<(), Box<dyn Error>> {
    let scene_name = args.scene.as_ref().unwrap();

    // Try to load JSON config file
    let config_file = if let Some(config_path) = &args.config_file {
        let json = json_helper::parse_json_from_path(config_path)?
            .as_object()
            .cloned()
            .ok_or_else(|| {
                format!(
                    "Expected a JSON object on the highest level in config file {}",
                    config_path.to_string_lossy()
                )
            })?;

        // Try to get section corresponding to the selected scene
        let scene_config = json.get(scene_name).ok_or_else(|| {
            format!(
                "Did not find entry for scene `{}` in config file `{}`",
                scene_name,
                config_path.to_string_lossy()
            )
        })?;

        Some(json_helper::JsonWrapper::new(scene_config.clone()))
    } else {
        None
    };

    // Build parameters to pass to scene constructor
    let scene_params = SceneParameters {
        output_dir: args.output_dir.join(scene_name).clone(),
        asset_dir: args.asset_dir.clone(),
        config_file,
    };

    info!("Starting to load scene {}.", scene_name);
    let mut scene = load_scene(&scene_name, &scene_params)?;

    assert_eq!(&scene.name, scene_name);
    info!("Loaded scene {}.", scene_name);

    if let Some(duration) = &args.duration {
        info!(
            "Overriding duration set by scene. Original: {}. New: {}.",
            scene.duration, *duration
        );
        scene.duration = *duration;
    }

    simulate_scene(scene, &args)?;

    coarse_prof_write_string()?
        .split("\n")
        .for_each(|l| info!("{}", l));
    info!("Simulation finished.");

    Ok(())
}

fn initialize_logging(args: &CommandlineArgs) -> Result<(), Box<dyn Error>> {
    // Try to load log filter level from env
    let mut unknown_log_filter_level = None;
    let log_filter_level = if let Some(log_level) = std::env::var_os("RUST_LOG") {
        let log_level = log_level.to_string_lossy().to_ascii_lowercase();
        match log_level.as_str() {
            "off" => log::LevelFilter::Off,
            "error" => log::LevelFilter::Error,
            "warn" => log::LevelFilter::Warn,
            "info" => log::LevelFilter::Info,
            "debug" => log::LevelFilter::Debug,
            "trace" => log::LevelFilter::Trace,
            _ => {
                unknown_log_filter_level = Some(log_level);
                log::LevelFilter::Info
            }
        }
    } else {
        // Default log level
        log::LevelFilter::Info
    };

    let log_dir = if let Some(scene_name) = &args.scene {
        args.output_dir.join(scene_name)
    } else {
        args.output_dir.clone()
    };
    fs::create_dir_all(&log_dir).map_err(|e| {
        format!(
            "Unable to create output directory '{}' ({:?})",
            log_dir.to_string_lossy(),
            e
        )
    })?;

    let log_file_path = if let Some(log_file_name) = &args.log_file {
        log_dir.join(log_file_name)
    } else {
        log_dir.join(format!(
            "femproto2_{}.log",
            chrono::Local::now().format("%F_%H-%M-%S-%6f")
        ))
    };

    let log_file = fs::OpenOptions::new()
        .write(true)
        .create(true)
        // TODO: To append or to truncate?
        .truncate(true)
        .append(false)
        .open(&log_file_path)
        .map_err(|e| {
            format!(
                "Unable to open log file '{}' for writing ({:?})",
                log_file_path.to_string_lossy(),
                e
            )
        })?;

    fern::Dispatch::new()
        .format(|out, message, record| {
            out.finish(format_args!(
                "[{}][{}][{}] {}",
                chrono::Local::now().to_rfc3339_opts(chrono::SecondsFormat::Micros, false),
                record.target(),
                record.level(),
                message
            ))
        })
        .level(log_filter_level)
        .chain(std::io::stdout())
        .chain(log_file)
        .apply()
        .map_err(|e| format!("Unable to apply logger configuration ({:?})", e))?;

    if let Some(filter_level) = unknown_log_filter_level {
        error!(
            "Unkown log filter level '{}' defined in 'RUST_LOG' env variable, using INFO instead.",
            filter_level
        );
    }

    Ok(())
}

fn initialize_thread_pool(args: &CommandlineArgs) -> Result<(), Box<dyn Error>> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(args.num_threads)
        .build_global()?;
    // TODO: Is there truly no way to determine the *actual* number of threads
    // in use by the global thread pool? Though, to be fair, it might be that a future version
    // of Rayon might have variable number of threads in the pool, so I suppose we are in
    // any case interested in the *maximum* number of threads in the pool.
    if args.num_threads == 0usize {
        if let Ok(rayon_num_threads) = env::var("RAYON_NUM_THREADS") {
            info!("Number of Rayon threads: {} (from environment)", rayon_num_threads);
        } else {
            info!("Number of Rayon threads: default");
        }
    } else {
        info!("Number of Rayon threads: {} (from command-line args)", args.num_threads);
    };

    Ok(())
}

fn print_git_info() -> Result<(), Box<dyn Error>> {
    info!("Build info");

    let build_timestamp = BUILD_TIMESTAMP.unwrap_or("Unknown");
    let build_hostname = BUILD_HOSTNAME.unwrap_or("Unknown");

    info!("\tBuild timestamp: {}", build_timestamp);
    info!("\tBuild hostname: '{}'", build_hostname);

    if let Some(last_commit) = GIT_LAST_COMMIT {
        info!("\tCommand 'git show -s --format=Commit: %H%nAuthor: %an, %aI%nTitle: '%s'':");
        for line in last_commit.split(";") {
            info!("\t\t{}", line);
        }
    } else {
        warn!("\tGit commit information was unavailable at build time");
    }

    if let Some(git_changes) = GIT_CHANGES {
        info!("\tCommand: 'git status -b --porcelain':");
        for line in git_changes.split(";") {
            info!("\t\t{}", line);
        }
    } else {
        warn!("\tGit file status information was unavailable at build time");
    }

    Ok(())
}

fn get_simulation_time(state: &StorageContainer) -> f64 {
    simulation_toolbox::components::get_simulation_time(state)
        .expect("SimulationTime must always be a component in the state.")
}

fn set_simulation_time(state: &mut StorageContainer, new_time: impl Into<SimulationTime>) {
    state.replace_storage(<SimulationTime as Component>::Storage::new(new_time.into()));
}

fn increment_step_index(state: &mut StorageContainer) {
    let step_index = state
        .try_get_component_storage::<StepIndex>()
        .expect("Simulation must have StepIndex component.")
        .borrow()
        .get_component()
        .clone();
    let new_step_index = StepIndex(step_index.0 + 1);
    state.replace_storage(<StepIndex as Component>::Storage::new(new_step_index));
}

fn increment_export_sequence_index(state: &mut StorageContainer) -> bool {
    let current_simulation_time = get_simulation_time(state);
    let current_dt = get_time_step(state).expect("Simulation must have TimeStep component.");

    let ExportSequenceIndex {
        index,
        prev_export_time,
        export_interval,
    } = state
        .try_get_component_storage::<ExportSequenceIndex>()
        .expect("Simulation must have ExportSequenceIndex component.")
        .borrow()
        .get_component()
        .clone();

    // TODO: Explicitly compute the StepIndex such that no StepIndex is closer for incrementing the ExportSequenceIndex

    // Elapsed simulation time since last export
    let elapsed = prev_export_time.map(|prev_export_time| current_simulation_time - prev_export_time);
    // Distance to reaching an export interval mark
    let delta_export = elapsed.map(|elapsed| (export_interval - elapsed).abs());
    // Check if distance is small enough to make an export
    let do_export = if let Some(delta_export) = delta_export {
        (elapsed.unwrap() > export_interval)
            || (delta_export < 100.0 * std::f64::EPSILON)
            || (delta_export < 0.01 * current_dt)
    } else {
        true
    };

    if do_export {
        let new_sequence_index = ExportSequenceIndex {
            index: index + 1,
            prev_export_time: Some(current_simulation_time),
            export_interval,
        };
        state.replace_storage(<ExportSequenceIndex as Component>::Storage::new(new_sequence_index));
    }

    do_export
}

pub fn new_output_sequence_filter_system<S>(system: S) -> impl System
where
    S: System,
{
    FilterSystem {
        predicate: {
            let mut last_export_index = None;
            move |state| {
                let current_export_index = get_export_sequence_index(state)?;
                if last_export_index.is_none() || current_export_index > last_export_index.unwrap() {
                    last_export_index = Some(current_export_index);
                    return Ok(true);
                }
                Ok(false)
            }
        },
        system,
    }
}

#[derive(Debug)]
struct DumpGlobalStashSystem {
    output_dir: PathBuf,
}

impl DumpGlobalStashSystem {
    pub fn new<P: Into<PathBuf>>(output_dir: P) -> Self {
        Self {
            output_dir: output_dir.into(),
        }
    }
}

impl Display for DumpGlobalStashSystem {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "GlobalStashDumper")
    }
}

impl System for DumpGlobalStashSystem {
    fn run(&mut self, data: &StorageContainer) -> Result<(), Box<dyn Error>> {
        let step_index = get_step_index(data)?;

        let stash_json = global_stash::to_string_pretty()?;
        fs::create_dir_all(&self.output_dir)?;
        let mut json_file = fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(
                self.output_dir
                    .join(format!("global_stash_{}.json", step_index)),
            )?;
        json_file.write_all(stash_json.as_bytes())?;

        global_stash::clear_values();
        Ok(())
    }
}

fn register_analysis_systems(systems: &mut Systems, args: &CommandlineArgs, scene_name: &str) {
    let output_dir = &args.output_dir;
    let vtk_basepath = output_dir.join(format!("{}/vtk/", scene_name));
    let ply_basepath = output_dir.join(format!("{}/ply/", scene_name));

    if !args.no_output_vtk {
        let vtk_systems = SystemCollection(vec![
            Box::new(VtkVolumeMeshOutput {
                base_path: vtk_basepath.clone(),
            }),
            Box::new(VtkSurfaceMeshOutput {
                base_path: vtk_basepath.clone(),
            }),
            Box::new(VtkFemOutput {
                base_path: vtk_basepath.clone(),
            }),
        ]);
        systems.add_system(Box::new(new_output_sequence_filter_system(TimingSystem::new(
            "VTK output",
            vtk_systems,
        ))));
    }

    if args.output_ply {
        let ply_systems = SystemCollection(vec![
            Box::new(PlyFem2dOutput {
                base_path: ply_basepath.clone(),
            }),
            Box::new(PlyVolumeMesh2dOutput {
                base_path: ply_basepath.clone(),
            }),
            Box::new(PlyInterpolatedPoints2dOutput {
                base_path: ply_basepath.clone(),
            }),
            Box::new(PlyPolyMesh3dOutput {
                base_path: ply_basepath.clone(),
            }),
            Box::new(PlyPolyMesh2dOutput {
                base_path: ply_basepath.clone(),
            }),
        ]);
        systems.add_system(Box::new(new_output_sequence_filter_system(TimingSystem::new(
            "PLY output",
            ply_systems,
        ))));
    }

    if args.dump_stash {
        systems.add_system(Box::new(DumpGlobalStashSystem::new(
            output_dir.join(format!("{}/stash/", scene_name)),
        )));
    }
}

fn simulate_scene(scene: Scene, args: &CommandlineArgs) -> Result<(), Box<dyn Error>> {
    register_component::<TimeStep>()?;
    register_component::<SimulationTime>()?;
    register_component::<Gravity>()?;
    register_component::<StepIndex>()?;

    let mut simulation_systems = scene.simulation_systems;
    let mut analysis_systems = scene.analysis_systems;
    let mut state = scene.initial_state;
    let duration = scene.duration;

    if let Some(dt) = args.timestep {
        state.replace_storage(<TimeStep as Component>::Storage::new(TimeStep(dt)));
    }

    let dt: f64 = if let Some(timestep) = state.try_get_component_storage::<TimeStep>() {
        timestep.borrow().get_component().clone().into()
    } else {
        let dt = 1.0 / 60.0;
        state.replace_storage(<TimeStep as Component>::Storage::new(TimeStep(dt)));
        dt
    };

    if state.try_get_component_storage::<TimeStep>().is_none() {
        // Set a default time step if the scene itself does not set one
        state.replace_storage(<TimeStep as Component>::Storage::new(TimeStep(1.0 / 60.0)));
    }

    if state.try_get_component_storage::<StepIndex>().is_none() {
        // If we have no StepIndex yet, start at 0
        state.replace_storage(<StepIndex as Component>::Storage::new(StepIndex(0)));
    }

    // Initialize the export sequence index
    {
        let export_interval = if let Some(fps) = args.output_fps {
            // Use frame time as export interval
            fps.recip()
        } else {
            0.0
        };

        state.replace_storage(<ExportSequenceIndex as Component>::Storage::new(ExportSequenceIndex {
            index: 0,
            prev_export_time: None,
            export_interval,
        }));
    }

    // Reset simulation time
    set_simulation_time(&mut state, 0.0);
    // Set gravity only if not set by the scene
    if !get_gravity(&mut state).is_ok() {
        set_gravity(&mut state, -9.81);
    }

    register_analysis_systems(&mut analysis_systems, &args, &scene.name);

    // Run analysis systems for initial state. Ensures output to VTK/PLY for initial state
    analysis_systems.run_all(&state)?;

    let mut progress_printer = ProgressPrinter::with_num_reports(100);

    info!("Starting simulation...");

    {
        profile!("simulation");
        stash_scope!("simulation");
        while get_simulation_time(&state) < duration {
            let do_export = {
                profile!("timestep");
                stash_scope!("timestep");
                increment_step_index(&mut state);
                let do_export = increment_export_sequence_index(&mut state);

                let step_index = state
                    .try_get_component_storage::<StepIndex>()
                    .unwrap()
                    .borrow()
                    .get_component()
                    .0;
                let t_before_sim = Instant::now();
                info!(
                    "Starting simulation of step {} @ time {:.8}s...",
                    step_index,
                    get_simulation_time(&state)
                );

                simulation_systems.run_all(&state)?;

                let new_time = get_simulation_time(&state) + dt;
                set_simulation_time(&mut state, new_time);
                analysis_systems.run_all(&state)?;

                // Print simulation timing
                let elapsed = t_before_sim.elapsed();
                let elapsed_s = (elapsed.as_secs() as f64) + (elapsed.subsec_nanos() as f64) * 1e-9;
                info!("Measured time for simulation of step {}: {:.6}s", step_index, elapsed_s);

                progress_printer.progress(new_time / duration);

                do_export
            };

            if args.print_timings_output && do_export && get_simulation_time(&state) < duration {
                coarse_prof_write_string()?
                    .split("\n")
                    .for_each(|l| info!("{}", l));
            }
        }
    }

    progress_printer.progress(1.0);

    Ok(())
}

struct ProgressPrinter {
    // Total number of reports to make
    num_reports: usize,
    // Previously reported progress, measured as integers in the interval [0, num_reports).
    current_progress: usize,
}

impl ProgressPrinter {
    fn with_num_reports(num_reports: usize) -> Self {
        Self {
            num_reports,
            current_progress: 0,
        }
    }

    fn progress(&mut self, progress_fraction: f64) {
        let quantizized = (progress_fraction * self.num_reports as f64).floor() as usize;
        // Possible off-by-one errors here, but whatever
        let quantitized = min(quantizized, self.num_reports);
        if quantitized > self.current_progress {
            self.current_progress = quantitized;
            info!(
                "Current progress: {} %.",
                (100.0 * self.current_progress as f64 / self.num_reports as f64)
            )
        }
    }
}

/// Returns the coarse_prof write output as a string
fn coarse_prof_write_string() -> Result<String, Box<dyn Error>> {
    let mut buffer = Vec::new();
    coarse_prof::write(&mut buffer)?;
    Ok(String::from_utf8_lossy(buffer.as_slice()).into_owned())
}
