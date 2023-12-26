# Condensed Summaries


## Getting Started
To use, your python version must be at least 3.11.

To get started, install the  package and its dependencies in editable mode by running:

```sh
pip install -e .
```

To run the tests, run:

```sh
pip install -e '.[test]'  # on Linux / MacOS
pip install -e ".[test]"  # on Windows
pytest ./tests
```


## compiling the C++ version

1. Download the latest boost libraries
2. Compile the libraries (needed for the commmand line argument parsing, see https://www.boost.org/doc/libs/1_83_0/more/getting_started/ )
3. Compile the bisumulation code:

    /usr/bin/g++ -std=c++20 -Wall -Wpedantic -Ofast -fdiagnostics-color=always -I ~/boost/boost_1_83_0_install/include/   full_bisimulation.cpp -o full_bisimulation  ~/boost/boost_1_83_0_install/lib/libboost_program_options.a 

4. Running on the command line:

./full_bisimulation run_timed mappingbased-objects_lang\=en.ttl

./full_bisimulation run_k_bisimulation_store_partition mappingbased-objects_lang\=en.ttl  --output=here.txt 

./full_bisimulation run_k_bisimulation_store_partition mappingbased-objects_lang\=en.ttl --k=3 --output=here.txt --skip_singletons --support=5



