ProjectAreas.csv

Every job is associated with a project ID (Area0 - Area60). This dataset provides the mapping of project ID to its domain science. For example, Area0 stands for Accelerator Physics which comes under the 'Physics' domain. Similarly, there are other areas, like Area3 (Astrophysics) and Area25 (Lattice Guage Theory) which are also classified under the Physics domain.

2015-2019.csv

These are the resource utilization datasets from the Titan supercompter at Oak Ridge National Laboratory for the years 2015-2019. Every file is associated with a header and each row corresponds to the total resource utilization statistics of a job that ran on Titan. The most important features are:
1. user_id: The used who submitted the job.
2. start_time: The timestamp when the job started.
3. end_time: The timestamp when the job ended.
4. stime, utime: The CPU time in Titan
5. alps_exit: status field - if a job runs successfully, only then job return status is zero
6. command: The executable of the job. The format is "/lustre/fileSystem/ProjectID-ProjectNumber/Application". For Titan, there were three different file system types under Lustre - atlas, atlas1 and atlas2. Jobs could select any one. 

Example: /lustre/atlas/User1027/Area8-7/acemd_2019.bin - The application 'acemd_2019.bin' belonged to project ID 'Area8' with the project Number '7'. User1027 submitted this job which ran on the 'atlas' file system in Lustre. 
Many commands do not have the file system or the user or the project information. 
Project ID is mapped to a domain science in the ProjectAreas.csv dataset. 

7. node_count: Number of nodes that the job ran.
8. max_rss: Estimate of the maximum resident CPU memory used by an individual compute node through the lifespan of a job run.
(Each Titan compute node is equipped with a 16-core CPU with a total of 32 GiB CPU memory, and every CPU is paired with a single GPU with a 6 GiB GPU memory.)
9. rchar, wchar: Bytes read and bytes written per process.
10. gpu_mode: Indicates how GPUs are used.  The GPU is in an exclusive mode, if only one process can operate a context to the GPU. An application can request the default mode, where multiple processes can communicate with a GPU.
11. gpu_secs: Time spent on GPUs by the job.
12. gpu_maxmem: Maximum GPU memory used by all the nodes.
13. gpu_summem: Total GPU memory used by all the nodes.

For better understanding of the RUR dataset, please read the paper:
[1] Feiyi Wang, Sarp Oral, Satyabrata Sen, and Neena Imam. "Learning from Five-year Resource-Utilization Data of Titan System." In 2019 IEEE International Conference on Cluster Computing (CLUSTER), pp. 1-6. IEEE, 2019.
