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
const int THREAD_NUM = 7;  
typedef struct
{
	int k;  
	int t_id;  
	int tasknum; 
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
//barrier定义
pthread_barrier_t barrier_Divsion;
pthread_barrier_t barrier_Elimination;
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
void GE(float** a, int n)
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
void C_GE(float** a, int n)
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
void C_GE_OMP_Dynamic(float** a, int n)
{  
	float t1, t2;  
	int i, j, k;
#pragma omp parallel num_threads(THREAD_NUM) shared(a) private(i, j, k, t1, t2)
	for (int k = 0; k < n; k++)
	{
		t1 = a[k][k];
#pragma omp single
		for (int j = k + 1; j < n; j++)
			a[k][j] = a[k][j] / t1;
		a[k][k] = 1.0;
#pragma omp for schedule(dynamic)
		for (int i = k + 1; i < n; i++)
		{
			t2 = a[i][k];
			for (int j = k + 1; j < n; j++)
				a[i][j] -= t2 * a[k][j];
			a[i][k] = 0;
		}
	}
}
// 高斯消去算法的Cache优化版本 _OMP
void C_GE_OMP_Static(float** a, int n)
{
	float t1, t2; 
	int i, j, k;
#pragma omp parallel num_threads(THREAD_NUM) shared(a) private(i, j, k, t1, t2)
	for (int k = 0; k < n; k++)
	{
		t1 = a[k][k];
#pragma omp single
		for (int j = k + 1; j < n; j++)
			a[k][j] = a[k][j] / t1;
		a[k][k] = 1.0;
#pragma omp for schedule(static)
		for (int i = k + 1; i < n; i++)
		{
			t2 = a[i][k];
			for (int j = k + 1; j < n; j++)
				a[i][j] -= t2 * a[k][j];
			a[i][k] = 0;
		}
	}
}
// 使用AVX指令集进行SIMD优化的高斯消去算法
void C_GE_OMP_Dynamic_AVX(float** a, int n)
{
	__m256 va, vt, vaik, vakj, vaij, vx;
	int i, j, k;
	float t1, t2;  
#pragma omp parallel num_threads(THREAD_NUM) shared(a) private(i, j, k, t1, t2, va, vt, vaik, vakj, vaij, vx)
	for (int k = 0; k < n; k++)
	{
		vt = _mm256_set1_ps(a[k][k]);
		t1 = a[k][k];
#pragma omp single
		{
			for (j = k + 1; j + 8 < n; j += 8)
			{
				va = _mm256_loadu_ps(&a[k][j]);
				va = _mm256_div_ps(va, vt);
				_mm256_storeu_ps(&a[k][j], va);
			}
			for (j; j < n; j++)
				a[k][j] = a[k][j] / t1;  
		}
#pragma omp barrier
		a[k][k] = 1.0;
#pragma omp for schedule(dynamic)
		for (int i = k + 1; i < n; i++)
		{
			vaik = _mm256_set1_ps(a[i][k]);
			t2 = a[i][k];
			for (j = k + 1; j + 8 < n; j += 8)
			{
				vakj = _mm256_loadu_ps(&a[k][j]);
				vaij = _mm256_loadu_ps(&a[i][j]);
				vx = _mm256_mul_ps(vakj, vaik);
				vaij = _mm256_sub_ps(vaij, vx);
				_mm256_storeu_ps(&a[i][j], vaij);
			}
			for (j; j < n; j++)
				a[i][j] -= t2 * a[k][j];
			a[i][k] = 0;
		}
#pragma omp barrier
	}
}
// 使用AVX指令集进行SIMD优化的高斯消去算法
void C_GE_OMP_Static_AVX(float** a, int n)
{
	__m256 va, vt, vaik, vakj, vaij, vx;
	int i, j, k;
	float t1, t2; 
#pragma omp parallel num_threads(THREAD_NUM) shared(a) private(i, j, k, t1, t2, va, vt, vaik, vakj, vaij, vx)
	for (int k = 0; k < n; k++)
	{
		vt = _mm256_set1_ps(a[k][k]);
		t1 = a[k][k];
#pragma omp single
		{
			for (j = k + 1; j + 8 < n; j += 8)
			{
				va = _mm256_loadu_ps(&a[k][j]);
				va = _mm256_div_ps(va, vt);
				_mm256_storeu_ps(&a[k][j], va);
			}
			for (j; j < n; j++)
				a[k][j] = a[k][j] / t1;  
		}
#pragma omp barrier
		a[k][k] = 1.0;
#pragma omp for schedule(static)
		for (int i = k + 1; i < n; i++)
		{
			vaik = _mm256_set1_ps(a[i][k]);
			t2 = a[i][k];
			for (j = k + 1; j + 8 < n; j += 8)
			{
				vakj = _mm256_loadu_ps(&a[k][j]);
				vaij = _mm256_loadu_ps(&a[i][j]);
				vx = _mm256_mul_ps(vakj, vaik);
				vaij = _mm256_sub_ps(vaij, vx);
				_mm256_storeu_ps(&a[i][j], vaij);
			}
			for (j; j < n; j++)
				a[i][j] -= t2 * a[k][j];
			a[i][k] = 0;
		}
#pragma omp barrier
	}
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
	time_use = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec);//微秒
	cout << "GE: " << time_use / 1000 << "ms" << endl;

	float** m2 = generate(n);
	gettimeofday(&start, NULL);
	C_GE(m2, n);
	gettimeofday(&end, NULL);
	time_use = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec);//微秒
	cout << "C_GE: " << time_use / 1000 << "ms" << endl;

	m2 = generate(n);
	gettimeofday(&start, NULL);
	C_GE_OMP_Dynamic(m2, n);
	gettimeofday(&end, NULL);
	time_use = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec);//微秒
	cout << "C_GE_OMP_Dynamic: " << time_use / 1000 << "ms" << endl;

	m2 = generate(n);
	gettimeofday(&start, NULL);
	C_GE_OMP_Static(m2, n);
	gettimeofday(&end, NULL);
	time_use = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec);//微秒
	cout << "C_GE_OMP_Static: " << time_use / 1000 << "ms" << endl;

	m2 = generate(n);
	gettimeofday(&start, NULL);
	C_GE_OMP_Dynamic_AVX(m2, n);
	gettimeofday(&end, NULL);
	time_use = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec);//微秒
	cout << "C_GE_OMP_Dynamic_AVX: " << time_use / 1000 << "ms" << endl;

	m2 = generate(n);
	gettimeofday(&start, NULL);
	C_GE_OMP_Static_AVX(m2, n);
	gettimeofday(&end, NULL);
	time_use = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec);//微秒
	cout << "C_GE_OMP_Static_AVX: " << time_use / 1000 << "ms" << endl;
}
