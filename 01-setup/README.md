# Setup instructions

Here’s how to set up CUDA on Ubuntu as a user, without sudo access:

1. **Download the CUDA Toolkit**:
    * Go to the [NVIDIA CUDA Toolkit Downloads page](https://developer.nvidia.com/cuda-downloads).
    * Select the appropriate options for your system (Linux, your architecture, Ubuntu, and the installer type - typically "runfile (local)").
    * Download the CUDA Toolkit `.run` file.
2. **Install CUDA Toolkit (without sudo)**:
    * Make the downloaded file executable:

```bash
chmod +x cuda_&lt;version&gt;_linux.run
```

    * Run the installer:

```bash
./cuda_&lt;version&gt;_linux.run --silent --toolkit --installpath=$HOME/cuda --override
```

    * The `--silent` option runs the installer without prompts.
    * `--toolkit` installs the CUDA toolkit only.
    * `--installpath=$HOME/cuda` specifies the installation directory in your home directory.  You can change `$HOME/cuda` to another location if you prefer.
    * `--override` overrides any previous installations.
    * **Important**:  If prompted about installing the driver, decline. You likely already have a driver installed on the system, and you typically need sudo rights to install drivers.  Let the system administrator handle driver updates.
3. **Set Environment Variables**:
    * Add the CUDA directories to your `PATH` and `LD_LIBRARY_PATH` environment variables.  Edit your `.bashrc` or `.zshrc` file (depending on which shell you use):

```bash
nano ~/.bashrc
```

    * Add the following lines to the end of the file, adjusting the path if you installed CUDA in a different location:

```bash
export PATH=$HOME/cuda/bin:$PATH
export LD_LIBRARY_PATH=$HOME/cuda/lib64:$LD_LIBRARY_PATH
```

    * Save the file and source it to apply the changes:

```bash
source ~/.bashrc
```

4. **Verify the Installation**:
    * Check the CUDA version:

```bash
nvcc --version
```

5. **Build and Run CUDA Samples (Optional)**:
    * Download the CUDA samples from NVIDIA (separate download).  You may need to register as a developer.  Choose the "CUDA Toolkit Samples" appropriate for your CUDA version.
    * Unzip the samples.
    * Navigate to the samples directory.
    * Modify the `Makefile` to reflect your CUDA installation path.  Specifically, look for the `CUDA_PATH` variable and set it to your installation directory (e.g., `CUDA_PATH := /home/your_username/cuda`).
    * Build the samples:

```bash
make
```

    * Run a sample (e.g., `deviceQuery`):

```bash
./bin/x86_64/linux/release/deviceQuery
```


## Important Considerations:

* **Driver Compatibility**:  This method assumes a compatible NVIDIA driver is already installed on the system. You are relying on the system administrator to maintain the driver.  Ensure that the CUDA Toolkit version you install is compatible with the installed driver. You can check the compatibility chart on NVIDIA's website.
* **System-Wide Impact**: Because you are installing CUDA locally in your home directory, it will only affect your user account. Other users on the system will not be affected unless they also modify their environment variables.
* **Conflicts**: Be aware of potential conflicts if the system administrator installs a different version of CUDA system-wide. Your locally installed version might take precedence depending on how the environment variables are set.
* **Updates**: You will be responsible for manually updating your CUDA installation.

If you encounter issues, carefully double-check the installation paths and environment variables. Also, consult the NVIDIA documentation for troubleshooting tips.

<div>⁂</div>

[^1]: https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/11514615/5816e6b0-8d5b-4ef4-ac81-9228944c8eb7/paste.txt

[^2]: https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/11514615/92082b26-d8fb-4727-a466-c13c9c9eeb63/paste.txt

[^3]: https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/11514615/e5c00516-9cb6-44b1-bc92-4222ba61b27b/paste.txt

