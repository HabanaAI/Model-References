# Steps to run gptj-99.9 Server

### Environment setup 
To setup the environment follow the steps described in [/closed/Intel-HabanaLabs/code/README.md](../../../code/README.md)

### Commands
Run the following commands from [/closed/Intel-HabanaLabs/code/](../../../code/) directory.

#### Run accuracy
```bash
source gptj-99.9/functions.sh
build_mlperf_inference --output-dir <output_dir> --submission gptj-99.9-fp8_Server --mode acc
```

#### Run performance
```bash
source gptj-99.9/functions.sh
build_mlperf_inference --output-dir <output_dir> --submission gptj-99.9-fp8_Server --mode perf
```

### Results

You can find the logs under /output_dir/logs/gptj-99.9-fp8/Server

For more details go to [/closed/Intel-HabanaLabs/code/README.md](../../../code/README.md)