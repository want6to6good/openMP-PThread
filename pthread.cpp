#include<iostream>
#include<stdlib.h>
#include<ctime>
#include<sys/time.h>
#include<immintrin.h>  // AVX
#include<pthread.h>  // pthread
#include<semaphore.h>  // �ź���
#include<mutex>
#include<condition_variable>
#include<omp.h>
#include<cmath>

using namespace std;

int n = 1000;
float** arr;
int lim = 1;
float** mat;
const int THREAD_NUM = 7;  // �߳�����

typedef struct
{
	int k;  //��ȥ���ִ�
	int t_id;  // �߳� id
	int tasknum;  // ��������
}PT_EliminationParam;
typedef struct
{
	int t_id; //�߳� id
}PT_StaticParam;
//�ź�������
sem_t sem_main;
sem_t sem_workerstart[THREAD_NUM]; 
sem_t sem_workerend[THREAD_NUM];
sem_t sem_leader;
sem_t sem_Divsion[THREAD_NUM - 1];
sem_t sem_Elimination[THREAD_NUM - 1];
//���������ͻ���������
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
//�̺߳���
void* PT_Block_Elimination(void* param)
{
	PT_EliminationParam* tt = (PT_EliminationParam*)param;
	int k = tt->k;
	int t_id = tt->t_id;
	int tasknum = tt->tasknum;
	int i = k + t_id * tasknum + 1;
	float temp;  // ���˼��ͬC_GE
	if (t_id != THREAD_NUM - 1)
	{
		for (int c = 0; c < tasknum; i++, c++)  // ִ�б��̶߳�Ӧ������c�����������
		{
			temp = mat[i][k];
			for (int j = k + 1; j < n; j++)
				mat[i][j] -= temp * mat[k][j];
			mat[i][k] = 0;
		}
	}
	else
	{
		for (; i < n; i++)  // ִ�б��̶߳�Ӧ������c�����������
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
	float temp;  // ���˼��ͬC_GE
	for (int i = k + t_id + 1; i < n; i += THREAD_NUM)  // ִ�б��̶߳�Ӧ������c�����������
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
	float t2;  // ʹ�ø������ݴ������Լ��ٳ����е�ַ�ķ��ʴ���
	for (int k = 0; k < n; k++)
	{
		sem_wait(&sem_workerstart[t_id]); // �������ȴ�������ɳ��������������Լ�ר�����ź�����
		//ѭ����������
		for (int i = k + 1 + t_id; i < n; i += THREAD_NUM)
		{
			//��ȥ
			t2 = mat[i][k];
			for (int j = k + 1; j < n; j++)
				mat[i][j] -= t2 * mat[k][j];
			mat[i][k] = 0.0;
		}
		sem_post(&sem_main); // �������߳�
		sem_wait(&sem_workerend[t_id]); //�������ȴ����̻߳��ѽ�����һ��
	}
	pthread_exit(nullptr);
	return nullptr;
}
void* PT_Static_Div_Elem(void* param)
{
	PT_StaticParam* p = (PT_StaticParam*)param;
	int t_id = p->t_id;
	float t1, t2;  // ʹ�ø������ݴ������Լ��ٳ����е�ַ�ķ��ʴ���
	for (int k = 0; k < n; ++k)
	{
		// t_id Ϊ 0 ���߳����������������������߳��ȵȴ�
		// ����ֻ������һ�������̸߳������������ͬѧ�ǿ��Գ��Բ��ö�������߳���ɳ�������
		// ���ź���������ͬ����ʽ��ʹ�� barrier
		if (t_id == 0)
		{
			for (int j = k + 1; j < n; j++)
				mat[k][j] = mat[k][j] / mat[k][k];
			mat[k][k] = 1.0;
		}
		else
			sem_wait(&sem_Divsion[t_id - 1]); // �������ȴ���ɳ�������
		// t_id Ϊ 0 ���̻߳������������̣߳�������ȥ����
		if (t_id == 0)
		{
			for (int i = 0; i < THREAD_NUM - 1; i++)
				sem_post(&sem_Divsion[i]);
		}

		//ѭ����������ͬѧ�ǿ��Գ��Զ������񻮷ַ�ʽ��
		for (int i = k + 1 + t_id; i < n; i += THREAD_NUM)
		{
			//��ȥ
			for (int j = k + 1; j < n; j++)
				mat[i][j] -= mat[i][k] * mat[k][j];
			mat[i][k] = 0.0;
		}
		if (t_id == 0)
		{
			for (int i = 0; i < THREAD_NUM - 1; i++)
				sem_wait(&sem_leader); // �ȴ����� worker �����ȥ
			for (int i = 0; i < THREAD_NUM - 1; i++)
				sem_post(&sem_Elimination[i]); // ֪ͨ���� worker ������һ��
		}
		else
		{
			sem_post(&sem_leader);// ֪ͨ leader, �������ȥ����
			sem_wait(&sem_Elimination[t_id - 1]); // �ȴ�֪ͨ��������һ��
		}
	}
	pthread_exit(nullptr);
	return nullptr;
}

void GE(float** a, int n)// ��׼�ĸ�˹��ȥ�㷨,
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
void C_GE(float** a, int n)// ��˹��ȥ�㷨��Cache�Ż��汾
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
void PThread_Dynamic_Block_GE(int n)// PThread�Ż��ĸ�˹��ȥ�㷨����̬�����黮������
{
	int tasknum = 0;
	pthread_t* handles = (pthread_t*)malloc((THREAD_NUM) * sizeof(pthread_t));  // Ϊ�߳̾�������ڴ�ռ�
	PT_EliminationParam* param = (PT_EliminationParam*)malloc((THREAD_NUM) * sizeof(PT_EliminationParam));  // �洢�̲߳���
	float t1, t2;  // ʹ�������������ݴ������Լ��ٳ����е�ַ�ķ��ʴ���
	for (int k = 0; k < n; k++)
	{
		t1 = mat[k][k];
		for (int j = k + 1; j < n; j++)
			mat[k][j] = mat[k][j] / t1;
		mat[k][k] = 1.0;
		//int thread_count = n - 1 - k;  // �����߳�����
		tasknum = (n - k - 1) / (THREAD_NUM - 1);  // ���߳���ִ�е����������������һ��Ϊ��Ϊ���ʣ�µ����������ռ�
		//��������
		if (tasknum < THREAD_NUM)  // ���һ���̵߳����������������߳������࣬��ô��ֱ�Ӳ������߳�
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
		{  // ���򴴽����߳�
			for (int t_id = 0; t_id < THREAD_NUM; t_id++)
			{
				//cout << k << " " << t_id << endl;
				param[t_id].k = k;
				param[t_id].t_id = t_id;
				param[t_id].tasknum = tasknum;
			}
			//�����߳�
			for (int t_id = 0; t_id < THREAD_NUM; t_id++)
				pthread_create(&handles[t_id], nullptr, PT_Block_Elimination, (void*)&param[t_id]);
			for (int t_id = 0; t_id < THREAD_NUM; t_id++)
				pthread_join(handles[t_id], nullptr);
		}
	}
}
void PThread_Dynamic_Rotation_GE(int n)// PThread�Ż��ĸ�˹��ȥ�㷨����̬��������������
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
void PThread_Static_OnlyElim_GE(int n)// PThread�Ż��ĸ�˹��ȥ�㷨����̬��ֻ������ѭ���еĺ���������߳�
{
	sem_init(&sem_main, 0, 0);//��ʼ���ź���
	for (int i = 0; i < THREAD_NUM; i++)
	{
		sem_init(&sem_workerstart[i], 0, 0);
		sem_init(&sem_workerend[i], 0, 0);
	}
	pthread_t handles[THREAD_NUM];// ������Ӧ�� Handle
	PT_StaticParam param[THREAD_NUM];// ������Ӧ���߳����ݽṹ
	float t1;  // ʹ�ø������ݴ������Լ��ٳ����е�ַ�ķ��ʴ���
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
		for (int t_id = 0; t_id < THREAD_NUM; t_id++)//��ʼ���ѹ����߳�
			sem_post(&sem_workerstart[t_id]);
		for (int t_id = 0; t_id < THREAD_NUM; t_id++)//���߳�˯�ߣ��ȴ����еĹ����߳���ɴ�����ȥ����
			sem_wait(&sem_main);	
		for (int t_id = 0; t_id < THREAD_NUM; t_id++)// ���߳��ٴλ��ѹ����߳̽�����һ�ִε���ȥ����
			sem_post(&sem_workerend[t_id]);
	}
	for (int t_id = 0; t_id < THREAD_NUM; t_id++)
		pthread_join(handles[t_id], nullptr);
	//���������ź���
	sem_destroy(&sem_main);
	sem_destroy(sem_workerstart);
	sem_destroy(sem_workerend);
}
void PThread_Static_GE(int n)// PThread�Ż��ĸ�˹��ȥ�㷨����̬��������ѭ��ȫ�������߳�

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