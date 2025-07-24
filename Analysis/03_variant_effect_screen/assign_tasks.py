import os
import sys

PYTHON = sys.executable

import click
import h5py
import numpy as np


def pick_tasks(h5_path, force=False):
    """Pick tasks from HDF5 storage"""
    with h5py.File(h5_path, "r") as f:
        if force:
            # Return all task indices
            return list(range(len(f["variants/index_key"])))
        else:
            # Return indices where status is 'pending' using boolean indexing (faster)
            statuses = f["variants/status"][:]
            pending_mask = statuses == "pending".encode()
            return np.where(pending_mask)[0].tolist()


def parse_compute_config(config_list):
    """Parse compute configurations from list of strings: machine:device_count:batch_size:device_type"""
    if not config_list:
        return []

    compute_assignments = []
    for config_str in config_list:
        parts = config_str.strip().split(':')
        if len(parts) != 4:
            raise ValueError(
                f"Invalid format: {config_str}. Expected: machine:device_count:batch_size:device_type"
            )

        machine, count_str, batch_size_str, device_type = parts
        try:
            machine = machine.strip()
            count = int(count_str.strip())
            batch_size = int(batch_size_str.strip())
            device_type = device_type.strip().lower()

            if device_type not in ['gpu', 'cpu']:
                raise ValueError(f"device_type must be 'gpu' or 'cpu', got: {device_type}")

            # Create one assignment per device
            for device_id in range(count):
                if device_type == 'gpu':
                    device = f'cuda:{device_id}'
                else:
                    device = 'cpu'

                compute_assignments.append({
                    'machine': machine,
                    'device': device,
                    'device_type': device_type,
                    'batch_size': batch_size
                })

        except ValueError as e:
            raise ValueError(f"Invalid values in {config_str}: {e}")

    return compute_assignments


def assign_tasks_to_compute(task_indices, compute_assignments):
    """Assign tasks to compute resources and update assignments in-place"""
    total_batch_capacity = sum(a['batch_size'] for a in compute_assignments)
    total_tasks = len(task_indices)
    start = 0
    
    for i, assignment in enumerate(compute_assignments):
        # Calculate chunk size proportional to batch size
        if i == len(compute_assignments) - 1:
            chunk_size = total_tasks - start  # Last chunk gets remaining tasks
        else:
            proportion = assignment['batch_size'] / total_batch_capacity
            chunk_size = int(total_tasks * proportion)
        
        end = start + chunk_size
        assignment['chunk_id'] = i
        assignment['task_indices'] = task_indices[start:end]
        start = end


@click.command()
@click.option("-h5", "--hdf5_file", required=True, help="Path to HDF5 file")
@click.option("-o", "--output_dir", required=True, help="Output directory for job files")
@click.option("-m", "--model_path", required=True, help="Path to packaged model file")
@click.option("-c", "--compute_script", default="compute.py", help="Path to compute script")
@click.option(
    "-g",
    "--compute_config",
    multiple=True,
    required=True,
    help="Compute configuration (format: machine:device_count:batch_size:device_type, device_type=gpu|cpu)",
)
@click.option("--force", is_flag=True, help="Force recompute all tasks")
def main(
    hdf5_file,
    output_dir,
    model_path,
    compute_script,
    compute_config,
    force,
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

    # Parse and assign tasks
    compute_assignments = parse_compute_config(compute_config)
    if not compute_assignments:
        print("Error: No compute configurations provided!")
        return

    assign_tasks_to_compute(task_indices, compute_assignments)

    # Track machine-level info for summary and scripts
    machine_chunks = {}
    total_tasks = 0

    # Generate scripts and collect info
    for assignment in compute_assignments:
        chunk_id = assignment['chunk_id']
        task_indices_chunk = assignment['task_indices']
        machine = assignment['machine']
        device = assignment['device']
        batch_size = assignment['batch_size']
        n_tasks = len(task_indices_chunk)
        total_tasks += n_tasks

        # Save task indices as npy file (more efficient)
        np.save(f"{output_dir}/chunk_{chunk_id}_indices.npy", np.array(task_indices_chunk))

        # Generate individual script
        script_path = f"{output_dir}/run_chunk_{chunk_id}.sh"
        with open(script_path, "w") as f:
            f.write(f"""#!/bin/bash
{PYTHON} {compute_script} \\
  --hdf5_file {hdf5_file} \\
  --chunk_indices {output_dir}/chunk_{chunk_id}_indices.npy \\
  --model_path {model_path} \\
  --device {device} \\
  --batch_size {batch_size}
""")
        os.chmod(script_path, 0o755)

        # Track machine info
        if machine not in machine_chunks:
            machine_chunks[machine] = []
        machine_chunks[machine].append(chunk_id)

    # Generate machine-level scripts
    for machine, chunk_ids in machine_chunks.items():
        script_path = f"{output_dir}/run_all_{machine}.sh"
        with open(script_path, "w") as f:
            f.write("#!/bin/bash\n")
            for chunk_id in chunk_ids:
                f.write(f"{output_dir}/run_chunk_{chunk_id}.sh &\n")
            f.write("wait\n")
        os.chmod(script_path, 0o755)

    # Detailed summary
    print(f"Generated {len(compute_assignments)} processes for {total_tasks} tasks:")
    for machine, chunk_ids in machine_chunks.items():
        machine_tasks = sum(len(compute_assignments[i]['task_indices']) for i in chunk_ids)
        print(f"  {machine}: {len(chunk_ids)} processes, {machine_tasks} tasks")
        for chunk_id in chunk_ids:
            assignment = compute_assignments[chunk_id]
            n_tasks = len(assignment['task_indices'])
            device = assignment['device']
            batch_size = assignment['batch_size']
            print(f"    Process {chunk_id} ({device}): {n_tasks} tasks, batch_size={batch_size}")


if __name__ == "__main__":
    main()
