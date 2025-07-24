import json
import os
import sys

PYTHON = sys.executable

import click
import h5py


def pick_tasks(h5_path, force=False):
    """Pick tasks from HDF5 storage"""
    with h5py.File(h5_path, "r") as f:
        if force:
            # Return all task indices
            return list(range(len(f["variants/index_key"])))
        else:
            # Return indices where status is 'pending'
            pending = "pending".encode()
            statuses = f["variants/status"][:]
            pending_indices = [i for i, status in enumerate(statuses) if status == pending]
            return pending_indices


def parse_gpu_machines(gpu_machines_str):
    """Parse GPU machine configuration from string format machine:gpu,machine:gpu,..."""
    if not gpu_machines_str:
        return {}
    
    machines = {}
    for machine_config in gpu_machines_str.split(','):
        if ':' not in machine_config:
            raise ValueError(f"Invalid machine format: {machine_config}. Expected format: machine:gpu_count")
        
        machine, gpu_count = machine_config.strip().split(':', 1)
        try:
            machines[machine.strip()] = int(gpu_count.strip())
        except ValueError:
            raise ValueError(f"Invalid GPU count for machine {machine}: {gpu_count}")
    
    return machines


def calculate_max_nprocs(gpu_machines, max_tasks_per_gpu=10):
    """Calculate maximum number of processes based on GPU resources"""
    if not gpu_machines:
        return None
    
    total_gpus = sum(gpu_machines.values())
    return total_gpus * max_tasks_per_gpu


def assign_chunks_to_gpus(nprocs, gpu_machines):
    """Assign chunk indices to GPU devices across machines"""
    if not gpu_machines:
        return None
    
    # Create list of all available GPUs
    gpu_list = []
    for machine, gpu_count in gpu_machines.items():
        for gpu_id in range(gpu_count):
            gpu_list.append({
                'machine': machine,
                'gpu_id': gpu_id,
                'device': f'cuda:{gpu_id}'
            })
    
    # Assign chunks to GPUs in round-robin fashion
    chunk_gpu_mapping = {}
    for chunk_id in range(nprocs):
        gpu_info = gpu_list[chunk_id % len(gpu_list)]
        chunk_gpu_mapping[chunk_id] = gpu_info
    
    return chunk_gpu_mapping


def chunkify(lst, k):
    """Divide list into k roughly equal chunks"""
    if not lst:
        return []

    avg = len(lst) // k
    remainder = len(lst) % k

    chunks = []
    start = 0

    for i in range(k):
        # Add one extra element to first 'remainder' chunks
        chunk_size = avg + (1 if i < remainder else 0)
        end = start + chunk_size

        if start < len(lst):
            chunks.append(lst[start:end])
        else:
            chunks.append([])

        start = end

    return chunks


def create_slurm_script(
    chunk_id, h5_path, model_path, compute_script_path, output_dir, device, slurm_config=None
):
    """Create SLURM script for a chunk"""

    script_content = f"""#!/bin/bash
#SBATCH --job-name=variant_effect_{chunk_id}
#SBATCH --partition={slurm_config['partition']}
#SBATCH --time={slurm_config['time']}
#SBATCH --cpus-per-task={slurm_config['cpus_per_task']}
#SBATCH --mem={slurm_config['mem']}
#SBATCH --output={output_dir}/slurm_%j.out
#SBATCH --error={output_dir}/slurm_%j.err

{PYTHON} {compute_script_path} \\
  --hdf5_file {h5_path} \\
  --chunk_file {output_dir}/chunk_{chunk_id}.json \\
  --model_path {model_path} \\
  --device {device}
"""

    script_path = f"{output_dir}/run_chunk_{chunk_id}.slurm"
    with open(script_path, "w") as f:
        f.write(script_content)

    return script_path


def create_machine_run_scripts(chunk_gpu_mapping, output_dir):
    """Create run_all.sh scripts for each machine by grouping individual chunk scripts"""
    machine_scripts = {}
    
    # Group chunks by machine
    machines = {}
    for chunk_id, gpu_info in chunk_gpu_mapping.items():
        machine = gpu_info['machine']
        if machine not in machines:
            machines[machine] = []
        machines[machine].append(chunk_id)
    
    # Create run script for each machine
    for machine, chunk_ids in machines.items():
        script_path = f"{output_dir}/run_all_{machine}.sh"
        
        with open(script_path, "w") as f:
            f.write("#!/bin/bash\n")
            f.write(f"# Run script for machine: {machine}\n")
            f.write(f"# Total chunks: {len(chunk_ids)}\n\n")
            
            # Add each chunk script as background process
            for chunk_id in chunk_ids:
                f.write(f"{output_dir}/run_chunk_{chunk_id}.sh &\n")
            
            f.write("\nwait\n")
            f.write(f'echo "All chunks completed on {machine}"\n')
        
        os.chmod(script_path, 0o755)
        machine_scripts[machine] = script_path
    
    return machine_scripts


@click.command()
@click.option("-h5", "--hdf5_file", required=True, help="Path to HDF5 file")
@click.option("-n", "--nprocs", type=int, required=True, help="Number of processes/jobs")
@click.option("-o", "--output_dir", required=True, help="Output directory for job files")
@click.option("-m", "--model_path", required=True, help="Path to packaged model file")
@click.option("-c", "--compute_script", default="compute.py", help="Path to compute script")
@click.option("--force", is_flag=True, help="Force recompute all tasks")
@click.option("--device", default="cpu", help="Computing device (cpu/cuda)")
@click.option("--gpu_machines", help="GPU machine configuration (format: machine1:gpu_count1,machine2:gpu_count2)")
@click.option("--max_tasks_per_gpu", type=int, default=10, help="Maximum tasks per GPU (default: 10)")
@click.option("--use_slurm", is_flag=True, help="Generate SLURM scripts")
@click.option("--slurm_partition", default="nova", help="SLURM partition")
@click.option("--slurm_time", default="0", help="SLURM time limit")
@click.option("--slurm_cpus", default=1, help="SLURM CPUs per task")
@click.option("--slurm_mem", default="8G", help="SLURM memory per task")
def main(
    hdf5_file,
    nprocs,
    output_dir,
    model_path,
    compute_script,
    force,
    device,
    gpu_machines,
    max_tasks_per_gpu,
    use_slurm,
    slurm_partition,
    slurm_time,
    slurm_cpus,
    slurm_mem,
):
    """Distribute variant effect computation tasks"""

    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.abspath(model_path)
    compute_script = os.path.abspath(compute_script)
    hdf5_file = os.path.abspath(hdf5_file)
    output_dir = os.path.abspath(output_dir)

    # Get task indices to process
    task_indices = pick_tasks(hdf5_file, force)
    print(f"Found {len(task_indices)} tasks to process")

    if not task_indices:
        print("No tasks to process!")
        return

    # Parse GPU machines if provided
    gpu_machine_config = parse_gpu_machines(gpu_machines) if gpu_machines else {}
    
    # Calculate max nprocs based on GPU resources if CUDA
    if device == "cuda" and gpu_machine_config:
        max_nprocs = calculate_max_nprocs(gpu_machine_config, max_tasks_per_gpu)
        if nprocs > max_nprocs:
            print(f"Warning: nprocs ({nprocs}) exceeds GPU capacity ({max_nprocs})")
            print(f"Capping nprocs to {max_nprocs}")
            nprocs = max_nprocs
        
        print(f"GPU resource summary:")
        total_gpus = sum(gpu_machine_config.values())
        print(f"  Total GPUs: {total_gpus}")
        print(f"  Max tasks per GPU: {max_tasks_per_gpu}")
        print(f"  Max processes: {max_nprocs}")
        print(f"  Using processes: {nprocs}")
    
    # Use original chunking logic
    chunks = chunkify(task_indices, nprocs)
    actual_chunks = [chunk for chunk in chunks if chunk]  # Remove empty chunks
    
    print(f"Divided into {len(actual_chunks)} non-empty jobs")
    
    # Save chunk information (same as original)
    for idx, chunk in enumerate(actual_chunks):
        chunk_data = {"chunk_id": idx, "task_indices": chunk, "n_tasks": len(chunk)}
        
        with open(f"{output_dir}/chunk_{idx}.json", "w") as fp:
            json.dump(chunk_data, fp, indent=2)
    
    # Assign chunks to GPUs if using CUDA
    chunk_gpu_mapping = None
    if device == "cuda" and gpu_machine_config:
        chunk_gpu_mapping = assign_chunks_to_gpus(len(actual_chunks), gpu_machine_config)
        print(f"GPU assignment:")
        for chunk_id, gpu_info in chunk_gpu_mapping.items():
            print(f"  Chunk {chunk_id} -> {gpu_info['machine']}:GPU{gpu_info['gpu_id']}")

    # Save overall job information
    job_info = {
        "total_tasks": len(task_indices),
        "n_chunks": len(actual_chunks),
        "hdf5_file": hdf5_file,
        "model_path": model_path,
        "compute_script": compute_script,
        "device": device,
        "force_recompute": force,
    }
    
    if device == "cuda" and gpu_machine_config:
        job_info["gpu_machines"] = gpu_machine_config
        job_info["max_tasks_per_gpu"] = max_tasks_per_gpu

    with open(f"{output_dir}/job_info.json", "w") as fp:
        json.dump(job_info, fp, indent=2)

    if use_slurm:
        # Generate SLURM scripts (unchanged for now)
        slurm_config = {
            "partition": slurm_partition,
            "time": slurm_time,
            "cpus_per_task": slurm_cpus,
            "mem": slurm_mem,
        }

        slurm_scripts = []
        for idx in range(len(actual_chunks)):
            script_path = create_slurm_script(
                chunk_id=idx,
                h5_path=hdf5_file,
                model_path=model_path,
                compute_script_path=compute_script,
                output_dir=output_dir,
                device=device,
                slurm_config=slurm_config,
            )
            slurm_scripts.append(script_path)

        # Create submission script
        submit_script = f"{output_dir}/submit_all.sh"
        with open(submit_script, "w") as f:
            f.write("#!/bin/bash\n\n")
            for script in slurm_scripts:
                f.write(f"sbatch {script}\n")

        os.chmod(submit_script, 0o755)

        print(f"Generated {len(slurm_scripts)} SLURM scripts")
        print(f"Run: {submit_script} to submit all jobs")

    else:
        # Generate individual chunk scripts (same as original)
        scripts = []
        for idx in range(len(actual_chunks)):
            script_path = f"{output_dir}/run_chunk_{idx}.sh"
            
            # Determine device for this chunk
            if chunk_gpu_mapping and idx in chunk_gpu_mapping:
                chunk_device = chunk_gpu_mapping[idx]['device']
            else:
                chunk_device = device
            
            with open(script_path, "w") as f:
                f.write(f"""#!/bin/bash
{PYTHON} {compute_script} \\
  --hdf5_file {hdf5_file} \\
  --chunk_file {output_dir}/chunk_{idx}.json \\
  --model_path {model_path} \\
  --device {chunk_device}
""")
            os.chmod(script_path, 0o755)
            scripts.append(script_path)

        print(f"Generated {len(scripts)} shell scripts for local execution")

        # Check if we have GPU machine configuration for machine-specific scripts
        if device == "cuda" and gpu_machine_config and chunk_gpu_mapping:
            # Generate machine-specific run scripts
            machine_scripts = create_machine_run_scripts(chunk_gpu_mapping, output_dir)
            
            print(f"Generated machine-specific run scripts:")
            for machine, script_path in machine_scripts.items():
                machine_chunks = [chunk_id for chunk_id, gpu_info in chunk_gpu_mapping.items() 
                                if gpu_info['machine'] == machine]
                total_tasks = sum(len(actual_chunks[chunk_id]) for chunk_id in machine_chunks)
                print(f"  {machine}: {script_path} ({len(machine_chunks)} chunks, {total_tasks} tasks)")
            
        else:
            # Create single run all script (original logic)
            run_all_script = f"{output_dir}/run_all.sh"
            with open(run_all_script, "w") as f:
                f.write("#!/bin/bash\n\n")
                for script in scripts:
                    f.write(f"{script} &\n")
                f.write("\nwait\necho \"All chunks completed\"\n")

            os.chmod(run_all_script, 0o755)
            print(f"Run: {run_all_script} to execute all chunks in parallel")

    print(f"\nTask distribution complete:")
    for i, chunk in enumerate(actual_chunks):
        print(f"  Chunk {i}: {len(chunk)} tasks")
    print(f"\nFiles saved to: {output_dir}")


if __name__ == "__main__":
    main()
