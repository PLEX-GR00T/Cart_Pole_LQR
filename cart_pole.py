import numpy as np
from numpy.core.fromnumeric import transpose
import random

M = 1
m = 0.3
b = 0.1
l = 1
I = 0.001
g = 9.8
dt = 0.01

Ra = -2
alpha = 0.01
noise = np.identity(4, dtype = float) * alpha

A = (I + (m*(l**2))*b)
B = (I*(M+m) + (M*m*(l)**2))
C = ((m**2)*g*(l**2))
D = (-m*l*b)
E = ((m*g*l)*(M+m))
F = (m*l)

AA = -A/B
BB = C/B
CC = D/B
DD = E/B

EE = A/B
FF = F/B

N = 500

Identity = np.identity(4, dtype = float)

Tb = np.array([
            [0, 1,  0,  0],
            [0, AA, BB, 0],
            [0, 0,  0,  1],
            [0, CC, DD, 0]]) 

Ta = np.array([0, EE, 0, FF]) * dt

Ts = Identity + (Tb)*dt

def s_(s,f):
    # S_desh = (Ts @ s) + (Ta*f)
    S_desh = (np.dot(Ts,s)) + (np.dot(Ta,f))
    return S_desh

force = [[]]*N
force[N-1] = 0.5
for i in range(N-2,-1,-1): 
    force[i] = random.randint(-2,2)
print("force", force[0])

S_desh = [[]]* N
S_desh[N-1] = [0.1, -0.05, 7, 0.1]
for i in range(N-2,-1,-1):
    S_desh[i] = s_(S_desh[i+1], force[i+1])
print("S_desh", S_desh[0])

new_s = [[]]*(N+1)
new_s[0] = S_desh[N-1]

rewd = [[]]*N
def Reward(rho):
    Rs =np.array([
            [rho/20, 0, 0, 0],
            [0, rho/20, 0, 0],
            [0,  0,   rho, 0],
            [0, 0, 0, rho/10]]) * (-1)
    
    rewd[N-1] = Rs
    print("Reward at N : ", rewd[N-1])
    print("Reward at 0 : ", rewd[0])
    print("------------------------------------------- before")
    for i in range(N-2,-1,-1):
        rewd[i]= np.dot(np.dot(S_desh[i].transpose(),Rs),S_desh[i]) + (force[i]*Ra*force[i])
    # r = (s.tranpose() @ Rs @ s) + (f.transpose() @ Ra @ f)
    return True

Qn = [[]]*N
Qn[N-1] = 0 

V = [[]]*N

print("V at N :", V[499])
F_lqr = [[]]*N
L_k = [[]]*N

def LQR():
    print("LQR Sdesh", S_desh[N-1])
    print("LQR Reward : ", rewd[N-1])
    V[N-1] = rewd[N-1]
    total_V = np.identity(4, dtype = float)

    for i in range(N-1, -1, -1):

        # print("Ts : ",Ts.transpose())
        # print("V[i]", V[i])
        # print("Dot pro : ",np.dot(Ts.transpose(), V[i]))
        a = np.dot(np.dot(Ts.transpose(), V[i]), Ts)
        # print("A :", a)

        b = np.dot(np.dot(Ta.transpose(), V[i]), Ts)
        # print("B :", b)
        
        c =  Ra + np.dot(np.dot(Ta.transpose(), V[i]), Ta)
        # print("C :", c)
        # d = np.dot(Ts, np.dot(Ta.transpose(), V[i]))

        V[i-1] = rewd[i] + a - (b.transpose())*(1/c) * b
        # print("LQR loop :", V[i])
        # print("total_V",total_V)
        total_V += V[i]
        # print("total_V",total_V)
        Qn[i-1] = Qn[i] + total_V.trace()

        
    for i in range(0, N, 1):
        a = Ra + np.dot(np.dot(Ta.transpose(), V[i]), Ta)
        b = np.dot(np.dot(Ta.transpose(), V[i]), Ts)

        L_k[i] = (1/a) * b 
        # print(L_k[i])
        # print("s desh", S_desh[i])
        F_lqr[i] = np.dot(-L_k[i], new_s[i])
        new_s[i+1] = np.dot(Ts, new_s[i]) + np.dot(Ta,F_lqr[i]) + noise



if __name__ == "__main__":
    if Reward(10): 
        LQR()
        print("Optimal policy : ", Qn[0])
        print("Optimal policy : ", Qn[0])

