import tempfile
import subprocess
import utils
import os

import sys
if 'torch' in sys.modules:
    print("[Warning] importing torch on the head node. Try to clean this up!")

DEFAULT_ENTRY_POINT = "scripts/train.py"
DEFAULT_REQUIRED_ARGS = ["save-path", "config"]

def parse_var(s):
    """
    Parse a key, value pair, separated by '='
    """
    items = s.split('=')
    key = items[0].strip() # we remove blanks around keys, as is logical
    if len(items) > 1:
        # rejoin the rest:
        value = '='.join(items[1:])
    return (key, value)

def parse_vars(items):
    """
    Parse a series of key-value pairs and return a dictionary
    """
    d = {}

    if items:
        for item in items:
            key, value = parse_var(item)
            d[key] = value
    return d

if __name__ == "__main__":

    parser = utils.get_slurm_parser()

    # Script arguments
    parser.add_argument("--entry-point", type=str, default=DEFAULT_ENTRY_POINT)
    parser.add_argument("--arguments", metavar="KEY=VALUE", nargs='+', help="Set kv pairs used as args for the entry point script.")
    parser.add_argument("--jobs-per-instance", type=int, default=1)

    args = parser.parse_args()
    script_args = parse_vars(args.arguments)

    # Handle the default case, train.
    if args.entry_point == DEFAULT_ENTRY_POINT:
        '''
        Custom code for sweeping using the experiment sweeper.
        '''
        for arg_name in DEFAULT_REQUIRED_ARGS:
            assert arg_name in script_args

        if script_args['config'].endswith(".json"):
            experiment = utils.Experiment.load(script_args['config'])
            configs_and_paths = [(c, os.path.join(script_args['save-path'], n)) for c, n in experiment.generate_configs_and_names()]
        else:
            configs_and_paths = [(script_args['config'], script_args['save-path'])]

        jobs = [{"config": c, "save-path" : p} for c, p in configs_and_paths]
        for arg_name in script_args.keys():
            if not arg_name in jobs[0]:
                print("Warning: argument", arg_name, "being added globally to all python calls with value", script_args[arg_name])
                for job in jobs:
                    job[arg_name] = script_args[arg_name]

    else:
        # we have the default configuration
        jobs = [script_args.copy() for _ in range(args.jobs_per_instance)]
        if args.jobs_per_instance:
            # We need some way of distinguishing the jobs, so set the seed argument
            # Scripts must implement this if they want to be able to run multiple on the save machine
            for i in range(args.jobs_per_instance):
                seed = jobs[i].get('seed', 0)
                jobs[i]['seed'] = int(seed) + i
    
    # Call python subprocess to launch the jobs.
    num_slurm_calls = len(jobs) // args.jobs_per_instance
    procs = []
    for i in range(num_slurm_calls):
        current_jobs = jobs[i*args.jobs_per_instance: (i+1)*args.jobs_per_instance]
        if i == num_slurm_calls - 1:
            current_jobs = jobs[i*args.jobs_per_instance:] # Get all the remaining jobs
        
        _, slurm_file = tempfile.mkstemp(text=True, prefix='job', suffix='.sh')
        print("Launching job with slurm configuration:", slurm_file)

        with open(slurm_file, 'w+') as f:
            utils.write_slurm_header(f, args)
            # Now that we have written the header we can launch the jobs.
            for job in current_jobs:
                command_str = ['python', args.entry_point]
                for arg_name, arg_value in job.items():
                    command_str.append("--" + arg_name)
                    command_str.append(str(arg_value))
                if len(current_jobs) != 1:
                    command_str.append('&')
                command_str = ' '.join(command_str) + '\n'
                f.write(command_str)
            if len(current_jobs) != 1:
                f.write('wait')
            
            # Now launch the job
            proc = subprocess.Popen(['sbatch', slurm_file])
            procs.append(proc)

    exit_codes = [p.wait() for p in procs]