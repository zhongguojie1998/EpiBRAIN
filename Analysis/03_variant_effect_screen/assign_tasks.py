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


@click.command()
@click.option("-h5", "--hdf5_file", required=True, help="Path to HDF5 file")
@click.option("-n", "--nprocs", type=int, required=True, help="Number of processes/jobs")
@click.option("-o", "--output_dir", required=True, help="Output directory for job files")
@click.option("-m", "--model_path", required=True, help="Path to packaged model file")
@click.option("-c", "--compute_script", default="compute.py", help="Path to compute script")
@click.option("--force", is_flag=True, help="Force recompute all tasks")
@click.option("--device", default="cpu", help="Computing device (cpu/cuda)")
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

    # Divide tasks into chunks
    chunks = chunkify(task_indices, nprocs)
    actual_chunks = [chunk for chunk in chunks if chunk]  # Remove empty chunks

    print(f"Divided into {len(actual_chunks)} non-empty jobs")

    # Save chunk information
    for idx, chunk in enumerate(actual_chunks):
        chunk_data = {"chunk_id": idx, "task_indices": chunk, "n_tasks": len(chunk)}

        # Save chunk file
        with open(f"{output_dir}/chunk_{idx}.json", "w") as fp:
            json.dump(chunk_data, fp, indent=2)

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

    with open(f"{output_dir}/job_info.json", "w") as fp:
        json.dump(job_info, fp, indent=2)

    if use_slurm:
        # Generate SLURM scripts
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
        # Generate simple shell scripts for local execution
        scripts = []
        for idx in range(len(actual_chunks)):
            script_path = f"{output_dir}/run_chunk_{idx}.sh"
            with open(script_path, "w") as f:
                f.write(
                    f"""#!/bin/bash
{PYTHON} {compute_script} \\
  --hdf5_file {hdf5_file} \\
  --chunk_file {output_dir}/chunk_{idx}.json \\
  --model_path {model_path} \\
  --device {device}
"""
                )
            os.chmod(script_path, 0o755)
            scripts.append(script_path)

        print(f"Generated {len(scripts)} shell scripts for local execution")

        # Create run all script (like submit_all.sh for SLURM)
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
