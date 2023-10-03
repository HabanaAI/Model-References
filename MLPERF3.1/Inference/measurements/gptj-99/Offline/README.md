# Steps to run gptj-99 Offline

### Environment setup 
To setup the environment follow the steps described in [/closed/Intel-HabanaLabs/code/README.md](../../../code/README.md)

### Commands
Run the following commands from [/closed/Intel-HabanaLabs/code/](../../../code/) directory.

#### Run accuracy
```bash
source gptj-99/functions.sh
build_mlperf_inference --output-dir <output_dir> --submission gptj-99-fp8_Offline --mode acc
```

#### Run performance
```bash
source gptj-99/functions.sh
build_mlperf_inference --output-dir <output_dir> --submission gptj-99-fp8_Offline --mode perf
```

### Results

You can find the logs under /output_dir/logs/gptj-99-fp8/Offline

For more details go to [/closed/Intel-HabanaLabs/code/README.md](../../../code/README.md)