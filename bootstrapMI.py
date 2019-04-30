import numpy as np
from scipy import stats
import math

#采用核密度估计的MI
def join_possibility_density(A,B):#A,B为一维数组

	len_A=len(A)
	len_B=len(B)

	A=A.reshape(len_A,1)
	B=(np.array(B)).reshape(len_B,1)
	A_B=np.hstack((A,B))
	values=A_B.T

	#向量维数
	d=2;N=len_A
	#用于MI的高斯核宽
	bandwidth=((4/(d+2))**(1/(d+4)))*(N**(-1/(d+4)))

	kde=stats.gaussian_kde(values, bw_method=bandwidth)
	density=kde(values)

	return density#返回一个数组

def marginal_possibility_density(x):

	len_x=len(x)

	values=x.T

	#向量维数
	d=1;N=len_x
	#用于MI的高斯核宽
	bandwidth=((4/(d+2))**(1/(d+4)))*(N**(-1/(d+4)))

	kde=stats.gaussian_kde(values, bw_method=bandwidth)
	density=kde(values)

	return density#返回一个数组

def MI(A,B):#A为一维数组，B为一维序列（预测值）

	assert len(A)==len(B)

	transfer=[join_possibility_density(A,B), marginal_possibility_density(A), marginal_possibility_density(B)]

	sumdata=[]
	for i in range(len(A)):
		sumdata.append(math.log((transfer[0][i])/((transfer[1][i])*(transfer[2][i]))))

	MI=(1/len(A))*sum(sumdata)

	return MI

def bootstrapMI(A, B, bRep=100, CI_Value=0.95):

	a=np.array(A)
	b=np.array(B)
	
	MI0 = MI(a,b)
	MI_transfer=[]
	for i in range(bRep):
		a=np.array(A)
		b=np.array(B)
	
		state = np.random.get_state()
		np.random.shuffle(a)
		np.random.set_state(state)
		np.random.shuffle(b)
		
		MI_transfer.append(MI(a,b))

	MI95=np.percentile(np.array(MI_transfer),CI_Value*100)

	return MI0, MI95, MI_transfer
