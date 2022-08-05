This repository is a product of a thesis titled Analysis and comparison of information retrieval systems.

Application's purpose is testing IR systems Apache Solr 8.11 and Milvus 2.0.2 on corpuses with with defined system configurations and specified queries.

## Requirements

- Downloaded project code
- Python 3.8.10
- Docker 
- CUDA supported GPU. [List](https://developer.nvidia.com/cuda-gpus)
- Anaconda Python distribution

## System installation

### Apache Solr
This system requires a build image for it's configurations. To build an image
got to the downloaded folder location and execute the following Docker command

```
$ docker build -t solr Solr
```

To verifiy build, you can list Docker images with 

```
$ docker images
```
Output

```
REPOSITORY            TAG                                 IMAGE ID       CREATED             SIZE
solr                  latest                              8fbd32ea38e6   About an hour ago   560MB
```
### Milvus
Milvus uses verified prebuild images downloaded with Docker Compose

## Start systems
To run systems containers use Docker Compose
```
$ docker-compose up -d
```
To verifiy execution, you can list running Docker containers with 

```
$ docker ps
```
Output 
```
CONTAINER ID   IMAGE                                           COMMAND                  CREATED         STATUS                            PORTS                      NAMES
ba85d2d8d37f   milvusdb/milvus:v2.0.2                          "/tini -- milvus run…"   5 seconds ago   Up 2 seconds                      0.0.0.0:19530->19530/tcp   milvus-standalone
3d87166e6def   postgres:14.1-alpine                            "docker-entrypoint.s…"   9 seconds ago   Up 4 seconds                      0.0.0.0:5438->5432/tcp     postgres
ec4371f02ae3   solr                                            "docker-entrypoint.s…"   9 seconds ago   Up 4 seconds                      0.0.0.0:8983->8983/tcp     solr
8b9c437a76dd   minio/minio:RELEASE.2022-07-08T00-05-23Z.fips   "/usr/bin/docker-ent…"   9 seconds ago   Up 4 seconds (health: starting)   9000/tcp                   milvus-minio
419e86f5c5e5   quay.io/coreos/etcd:v3.5.0                      "etcd -advertise-cli…"   9 seconds ago   Up 4 seconds                      2379-2380/tcp              milvus-etcd
```
## Stop systems 
```
$ docker-compose down
```
It's recommended to clear docker volumes and virtual memory if on Windows.

## Preparing testing enviroment
Project dependencies are  installed in Anaconda Python enviroment.

### Conda enviroment
To create an enviroment with specified Python version execute the following command in Anaconda promt
```
> conda create -n <name> python=3.8.10
```
Verify with 
```
> conda env list
```
To activate new enviroment do
```
> activate <name>
```

### Cuda
To enable GPU with CUDA, CudaToolkit must be [downloaded](https://developer.nvidia.com/cuda-11-3-1-download-archive) and installed in this enviroment.
```
> conda install numba & conda install cudatoolkit
```

App requires Pytorch 1.11.0 library, which must be [downloaded](https://pytorch.org/get-started/locally/) and installed with support for CUDA.
```
> conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
```
### Other dependencies
The rest of project's dependencies are installed via the following pip command. This requires you to be in a downloaded projects location, where requirements.txt is located.
```
python -m pip install -r requirements.txt
```

## Running tests
To run system test you must go to /App directory inside downloaded projects's directory.. There is a Python script called run_config_tests.py, that runs test commands specified with handlers. 
```
> python run_config_tests.py -h
  ---
  Script runs tests for selected system and configuration.
 -s  --System      System to be tested. [Milvus/Solr]
 -t  --Test        Test case to be tested. [1-3]
 -a  --Action      Action to be executed. If none specified index and query are selected. Query option requires already indexed data. [index/query/purge]
 -i  --Iteration   Number of tests. Default is one, which is recommended for indexing of larger files.
 -d  --Drop        Drop table and collections, enabled by default. This only works with Milvus system. [0/1]
 -c  --Clear       Clear data, enabled by default.  [0/1]
 -f  --File        Specifiy absolut path for the log file. Default directory is in App/logs/.
```
### Test examples
1. Index and query data
```
> python run_config_tests.py -s solr -t 1
```
Excpected output
```
--------------------------------------------------
|                TESTING STARTED                 |
--------------------------------------------------

--------------------------------------------------
|                  TESTING SOLR                  |
--------------------------------------------------

Test 1.
-> initializing system
-> indexing documents
██████████████████████████████████████████████████ [40000/40000]
-> querying documents
██████████████████████████████████████████████████ [12/12]
-> clearing data


--------------------------------------------------
|                 TESTING ENDED                  |
--------------------------------------------------
Results in: logs/05-08-2022.log

```

2. Test indexing without clearing 
```
> python run_config_tests.py -s milvus -t 1 -a index -c 0 -d 0
```

3. Run multiple test on already indexed data
```
> python run_config_tests.py -s milvus -t 1 -a query -i 3 -c 0 -d 0
```

4. Clear indexed data after testing
```
> python run_config_tests.py -s solr -t 3 -a purge
```
