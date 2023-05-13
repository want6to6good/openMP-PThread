#include<iostream>
#include<stdlib.h>
#include<ctime>
#include<sys/time.h>
#include<immintrin.h>  // AVX
#include<pthread.h>  // pthread
#include<semaphore.h>  // 信号量
#include<mutex>
#include<condition_variable>
#include<omp.h>
#include<cmath>

using namespace std;

int n = 1000;
float** arr;
int lim = 1;
float** mat;
const int THREAD_NUM = 7;  // 线程数量

typedef struct
{
	int k;  //消去的轮次
	int t_id;  // 线程 id
	int tasknum;  // 任务数量
}PT_EliminationParam;
typedef struct
{
	int t_id; //线程 id
}PT_StaticParam;
//信号量定义
sem_t sem_main;
sem_t sem_workerstart[THREAD_NUM]; 
sem_t sem_workerend[THREAD_NUM];
sem_t sem_leader;
sem_t sem_Divsion[THREAD_NUM - 1];
sem_t sem_Elimination[THREAD_NUM - 1];
//条件变量和互斥锁定义
mutex _mutex;
condition_variable _cond;
void init_()
{
	mat = new float* [n];
	arr = new float* [n];
	for (int i = 0; i < n; i++)
	{
		mat[i] = new float[n];
		arr[i] = new float[n];
		for (int j = 0; j < n; j++)
		{
			arr[i][j] = rand() % 10;
		}
	}
}
float** generate()
{
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
			mat[i][j] = arr[i][j];
	}
	return mat;
}
//线程函数
void* PT_Block_Elimination(void* param)
{
	PT_EliminationParam* tt = (PT_EliminationParam*)param;
	int k = tt->k;
	int t_id = tt->t_id;
	int tasknum = tt->tasknum;
	int i = k + t_id * tasknum + 1;
	float temp;  // 设计思想同C_GE
	if (t_id != THREAD_NUM - 1)
	{
		for (int c = 0; c < tasknum; i++, c++)  // 执行本线程对应的任务，c代表任务计数
		{
			temp = mat[i][k];
			for (int j = k + 1; j < n; j++)
				mat[i][j] -= temp * mat[k][j];
			mat[i][k] = 0;
		}
	}
	else
	{
		for (; i < n; i++)  // 执行本线程对应的任务，c代表任务计数
		{
			temp = mat[i][k];
			for (int j = k + 1; j < n; j++)
				mat[i][j] -= temp * mat[k][j];
			mat[i][k] = 0;
		}
	}
	pthread_exit(nullptr);
	return nullptr;
}
void* PT_Rotation_Elimination(void* param)
{
	PT_EliminationParam* tt = (PT_EliminationParam*)param;
	int k = tt->k;
	int t_id = tt->t_id;
	float temp;  // 设计思想同C_GE
	for (int i = k + t_id + 1; i < n; i += THREAD_NUM)  // 执行本线程对应的任务，c代表任务计数
	{
		temp = mat[i][k];
		for (int j = k + 1; j < n; j++)
			mat[i][j] -= temp * mat[k][j];
		mat[i][k] = 0;
	}
	pthread_exit(nullptr);
	return nullptr;
}
void* PT_Static_Elimination(void* param)
{
	PT_StaticParam* p = (PT_StaticParam*)param;
	int t_id = p->t_id;
	float t2;  // 使用浮点数暂存数据以减少程序中地址的访问次数
	for (int k = 0; k < n; k++)
	{
		sem_wait(&sem_workerstart[t_id]); // 阻塞，等待主线完成除法操作（操作自己专属的信号量）
		//循环划分任务
		for (int i = k + 1 + t_id; i < n; i += THREAD_NUM)
		{
			//消去
			t2 = mat[i][k];
			for (int j = k + 1; j < n; j++)
				mat[i][j] -= t2 * mat[k][j];
			mat[i][k] = 0.0;
		}
		sem_post(&sem_main); // 唤醒主线程
		sem_wait(&sem_workerend[t_id]); //阻塞，等待主线程唤醒进入下一轮
	}
	pthread_exit(nullptr);
	return nullptr;
}
void* PT_Static_Div_Elem(void* param)
{
	PT_StaticParam* p = (PT_StaticParam*)param;
	int t_id = p->t_id;
	float t1, t2;  // 使用浮点数暂存数据以减少程序中地址的访问次数
	for (int k = 0; k < n; ++k)
	{
		// t_id 为 0 的线程做除法操作，其它工作线程先等待
		// 这里只采用了一个工作线程负责除法操作，同学们可以尝试采用多个工作线程完成除法操作
		// 比信号量更简洁的同步方式是使用 barrier
		if (t_id == 0)
		{
			for (int j = k + 1; j < n; j++)
				mat[k][j] = mat[k][j] / mat[k][k];
			mat[k][k] = 1.0;
		}
		else
			sem_wait(&sem_Divsion[t_id - 1]); // 阻塞，等待完成除法操作
		// t_id 为 0 的线程唤醒其它工作线程，进行消去操作
		if (t_id == 0)
		{
			for (int i = 0; i < THREAD_NUM - 1; i++)
				sem_post(&sem_Divsion[i]);
		}

		//循环划分任务（同学们可以尝试多种任务划分方式）
		for (int i = k + 1 + t_id; i < n; i += THREAD_NUM)
		{
			//消去
			for (int j = k + 1; j < n; j++)
				mat[i][j] -= mat[i][k] * mat[k][j];
			mat[i][k] = 0.0;
		}
		if (t_id == 0)
		{
			for (int i = 0; i < THREAD_NUM - 1; i++)
				sem_wait(&sem_leader); // 等待其它 worker 完成消去
			for (int i = 0; i < THREAD_NUM - 1; i++)
				sem_post(&sem_Elimination[i]); // 通知其它 worker 进入下一轮
		}
		else
		{
			sem_post(&sem_leader);// 通知 leader, 已完成消去任务
			sem_wait(&sem_Elimination[t_id - 1]); // 等待通知，进入下一轮
		}
	}
	pthread_exit(nullptr);
	return nullptr;
}

void GE(float** a, int n)// 标准的高斯消去算法,
{
	for (int k = 0; k < n; k++)
	{
		for (int j = k + 1; j < n; j++)
			a[k][j] = a[k][j] / a[k][k];
		a[k][k] = 1.0;
		for (int i = k + 1; i < n; i++)
		{
			for (int j = k + 1; j < n; j++)
				a[i][j] -= a[i][k] * a[k][j];
			a[i][k] = 0;
		}
	}
}
void C_GE(float** a, int n)// 高斯消去算法的Cache优化版本
{
	float t1, t2; 
	for (int k = 0; k < n; k++)
	{
		t1 = a[k][k];
		for (int j = k + 1; j < n; j++)
			a[k][j] = a[k][j] / t1;
		a[k][k] = 1.0;
		for (int i = k + 1; i < n; i++)
		{
			t2 = a[i][k];
			for (int j = k + 1; j < n; j++)
				a[i][j] -= t2 * a[k][j];
			a[i][k] = 0;
		}
	}
}
void PThread_Dynamic_Block_GE(int n)// PThread优化的高斯消去算法，动态，按块划分任务
{
	int tasknum = 0;
	pthread_t* handles = (pthread_t*)malloc((THREAD_NUM) * sizeof(pthread_t));  // 为线程句柄分配内存空间
	PT_EliminationParam* param = (PT_EliminationParam*)malloc((THREAD_NUM) * sizeof(PT_EliminationParam));  // 存储线程参数
	float t1, t2;  // 使用两个浮点数暂存数据以减少程序中地址的访问次数
	for (int k = 0; k < n; k++)
	{
		t1 = mat[k][k];
		for (int j = k + 1; j < n; j++)
			mat[k][j] = mat[k][j] / t1;
		mat[k][k] = 1.0;
		//int thread_count = n - 1 - k;  // 设置线程数量
		tasknum = (n - k - 1) / (THREAD_NUM - 1);  // 本线程中执行的任务数量，将其减一是为了为最后剩下的任务留出空间
		//分配任务
		if (tasknum < THREAD_NUM)  // 如果一个线程的任务数量还不如线程数量多，那么就直接采用主线程
		{
			for (int i = k + 1; i < n; i++)
			{
				t2 = mat[i][k];
				for (int j = k + 1; j < n; j++)
					mat[i][j] -= t2 * mat[k][j];
				mat[i][k] = 0;
			}
		}
		else
		{  // 否则创建多线程
			for (int t_id = 0; t_id < THREAD_NUM; t_id++)
			{
				//cout << k << " " << t_id << endl;
				param[t_id].k = k;
				param[t_id].t_id = t_id;
				param[t_id].tasknum = tasknum;
			}
			//创建线程
			for (int t_id = 0; t_id < THREAD_NUM; t_id++)
				pthread_create(&handles[t_id], nullptr, PT_Block_Elimination, (void*)&param[t_id]);
			for (int t_id = 0; t_id < THREAD_NUM; t_id++)
				pthread_join(handles[t_id], nullptr);
		}
	}
}
void PThread_Dynamic_Rotation_GE(int n)// PThread优化的高斯消去算法，动态，轮流划分任务
{
	int tasknum = 0;
	pthread_t* handles = (pthread_t*)malloc(THREAD_NUM * sizeof(pthread_t));  
	PT_EliminationParam* param = (PT_EliminationParam*)malloc(THREAD_NUM * sizeof(PT_EliminationParam));  
	float t1, t2;  
	for (int k = 0; k < n; k++)
	{
		t1 = mat[k][k];
		for (int j = k + 1; j < n; j++)
			mat[k][j] = mat[k][j] / t1;
		mat[k][k] = 1.0;
		tasknum = (n - k - 1) / (THREAD_NUM);  
		if (tasknum < THREAD_NUM)  
		{
			for (int i = k + 1; i < n; i++)
			{
				t2 = mat[i][k];
				for (int j = k + 1; j < n; j++)
					mat[i][j] -= t2 * mat[k][j];
				mat[i][k] = 0;
			}
		}
		else
		{
			for (int t_id = 0; t_id < THREAD_NUM; t_id++)
			{
				//cout << k << " " << t_id << endl;
				param[t_id].k = k;
				param[t_id].t_id = t_id;
			}
			for (int t_id = 0; t_id < THREAD_NUM; t_id++)
				pthread_create(&handles[t_id], nullptr, PT_Rotation_Elimination, (void*)&param[t_id]);
			for (int t_id = 0; t_id < THREAD_NUM; t_id++)
				pthread_join(handles[t_id], nullptr);
		}
	}
}
void PThread_Static_OnlyElim_GE(int n)// PThread优化的高斯消去算法，静态，只将三重循环中的后二重纳入线程
{
	sem_init(&sem_main, 0, 0);//初始化信号量
	for (int i = 0; i < THREAD_NUM; i++)
	{
		sem_init(&sem_workerstart[i], 0, 0);
		sem_init(&sem_workerend[i], 0, 0);
	}
	pthread_t handles[THREAD_NUM];// 创建对应的 Handle
	PT_StaticParam param[THREAD_NUM];// 创建对应的线程数据结构
	float t1;  // 使用浮点数暂存数据以减少程序中地址的访问次数
	for (int t_id = 0; t_id < THREAD_NUM; t_id++)
	{
		param[t_id].t_id = t_id;
		pthread_create(&handles[t_id], nullptr, PT_Static_Elimination, (void*)&param[t_id]);
	}
	for (int k = 0; k < n; k++)
	{
		t1 = mat[k][k];
		for (int j = k + 1; j < n; j++)
			mat[k][j] /= t1;
		mat[k][k] = 1.0;
		for (int t_id = 0; t_id < THREAD_NUM; t_id++)//开始唤醒工作线程
			sem_post(&sem_workerstart[t_id]);
		for (int t_id = 0; t_id < THREAD_NUM; t_id++)//主线程睡眠（等待所有的工作线程完成此轮消去任务）
			sem_wait(&sem_main);	
		for (int t_id = 0; t_id < THREAD_NUM; t_id++)// 主线程再次唤醒工作线程进入下一轮次的消去任务
			sem_post(&sem_workerend[t_id]);
	}
	for (int t_id = 0; t_id < THREAD_NUM; t_id++)
		pthread_join(handles[t_id], nullptr);
	//销毁所有信号量
	sem_destroy(&sem_main);
	sem_destroy(sem_workerstart);
	sem_destroy(sem_workerend);
}
void PThread_Static_GE(int n)// PThread优化的高斯消去算法，静态，将三重循环全部纳入线程

{
	sem_init(&sem_leader, 0, 0);
	for (int i = 0; i < THREAD_NUM - 1; ++i)
	{
		sem_init(&sem_Divsion[i], 0, 0);
		sem_init(&sem_Elimination[i], 0, 0);
	}
	pthread_t handles[THREAD_NUM];
	PT_StaticParam param[THREAD_NUM];
	for (int t_id = 0; t_id < THREAD_NUM; t_id++)
	{
		param[t_id].t_id = t_id;
		pthread_create(&handles[t_id], nullptr, PT_Static_Div_Elem, (void*)&param[t_id]);
	}
	for (int t_id = 0; t_id < THREAD_NUM; t_id++)
		pthread_join(handles[t_id], nullptr);
	sem_destroy(&sem_leader);
	sem_destroy(sem_Divsion);
	sem_destroy(sem_Elimination);
}

int main()
{
	srand(time(0));
	cin >> n;
	init_();
	struct timeval start, end;
	float time_use = 0;

	float** m1 = generate(n);
	gettimeofday(&start, NULL);
	GE(m1, n);
	gettimeofday(&end, NULL);
	cout << endl << endl << endl;
	time_use = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec);
	cout << "GE: " << time_use / 1000 << "ms" << endl;

	float** m2 = generate(n);
	gettimeofday(&start, NULL);
	C_GE(m2, n);
	gettimeofday(&end, NULL);
	time_use = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec);
	cout << "C_GE: " << time_use / 1000 << "ms" << endl;

	generate(n);
	gettimeofday(&start, NULL);
	PThread_Dynamic_Block_GE(n);
	gettimeofday(&end, NULL);
	time_use = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec);
	cout << "PThread_Dynamic_Block_GE: " << time_use / 1000 << "ms" << endl;

	generate(n);
	gettimeofday(&start, NULL);
	PThread_Dynamic_Rotation_GE(n);
	gettimeofday(&end, NULL);
	time_use = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec);
	cout << "PThread_Dynamic_Rotation_GE: " << time_use / 1000 << "ms" << endl;

	generate(n);
	gettimeofday(&start, NULL);
	PThread_Static_OnlyElim_GE(n);
	gettimeofday(&end, NULL);
	time_use = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec);
	cout << "PThread_Static_OnlyElim_GE: " << time_use / 1000 << "ms" << endl;

	generate(n);
	gettimeofday(&start, NULL);
	PThread_Static_GE(n); 
	gettimeofday(&end, NULL);
	time_use = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec);
	cout << "PThread_Static_GE: " << time_use / 1000 << "ms" << endl;
}