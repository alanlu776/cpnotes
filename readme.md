# Competitive Programming Notes

## Code

Disjoint Set
```
n=int(input())
A=[[1,1] for i in range(n+1)]
def find(x):
    if A[x][1]>0: return x
    return find(-A[x][0])
def connect(a,b):
    a,b=find(a),find(b)
    if A[a][1]<A[b][1]:
        c=a
        a=b
        b=c
    A[a][0]+=A[b][0]
    A[a][1]=max(A[a][1],A[b][1]+1)
    A[b][0]=-a
    A[b][1]=0
for i in range(n-1):
    a,b=map(int,input().split())
    connect(a,b)
```
Modulo Inverses
```
MOD=1000000007
def power(x,y,m):
    if y==0:
        return 1
    p=power(x,y//2,m)%m
    p=(p*p)%m
    if y%2==0:
        return p
    else:
        return (x*p)%m
inverses=[-1]*1000000
def inverse(a,m):
    if inverses[a]!=-1:
        return inverses[a]
    inverses[a]=power(a,m-2,m)
    return inverses[a]
for _ in range(int(input())):
    m,n=map(int,input().split())
    answer=1
    for i in range(1,m+n-1):
        answer=(answer*i)%MOD
    for i in range(1,m):
        answer=(answer*inverse(i,MOD))%MOD
    for i in range(1,n):
        answer=(answer*inverse(i,MOD))%MOD
    print(answer)
```
Modulo Inverses 2
```
MOD=1000000007
F=[1]+[0]*50000
for i in range(50000):
    F[i+1]=F[i]*(i+1)%MOD
def power(x,y,m):
    if y==0:return 1
    p=power(x,y//2,m)%m
    p=(p*p)%m
    if y%2==0:return p
    else:return (x*p)%m
IF=[1]+[-1]*50000
def inverseFactorial(a,m):
    if IF[a]!=-1:return IF[a]
    IF[a]=power(F[a],m-2,m)
    return IF[a]
s=input()
sums=[[0]*(len(s)+1) for _ in range(26)]
for i in range(len(s)):
    for j in range(26):
        sums[j][i+1]=sums[j][i]
        if s[i]==chr(ord('a')+j):sums[j][i+1]+=1
for _ in range(int(input())):
    l,r=map(int,input().split())
    nums=[sums[i][r]-sums[i][l-1] for i in range(26)]
    last=sum([nums[i]%2==1 for i in range(26)])
    for i in range(26):nums[i]//=2
    answer=F[sum(nums)]
    for i in range(26):answer=answer*inverseFactorial(nums[i],MOD)%MOD
    if answer==0:answer=1
    if last:answer=answer*last%MOD
    print(answer)
```
Modulo Inverses wtf
```
def solve(n, m):
    inv=lambda a,b,c,d:(b<2)*d or inv(b,a%b,d,c-a//b*d)
    p,q,C=1,1,10**9+7
    for i in range(m,n+m-1):
        p,q=p*i%C,q*(i-m+1)%C
    return p*inv(q,C,1,0)%C
```
Segment Tree
```
n=10
tree=[0]*(2*n)
def build(arr):
    for i in range(n):
        tree[n+i]=arr[i]
    for i in range(n-1,0,-1):
        tree[i]=tree[i<<1]+tree[i<<1|1]
def update(p,value):
    tree[p+n]=value
    i=p+n
    while i>1:
        tree[i>>1]=tree[i]+tree[i^1]
        i>>=1
def query(l,r):
    res=0
    l,r=l+n,r+n
    while l<r:
        if (l&1):
            res+=tree[l]
            l+=1
        if (r&1):
            r-=1
            res+=tree[r]
        l>>=1
        r>>=1
    return res
```
```
#include <bits/stdc++.h>
using namespace std;
 
const int N = 100000;
int n;
int tree[2 * N];

void build( int arr[])
{
    for (int i=0; i<n; i++)   
        tree[n+i] = arr[i];
    for (int i=n-1; i>0; --i)    
        tree[i] = tree[i<<1] + tree[i<<1|1];   
}

void updateTreeNode(int p, int value) {
    tree[p+n] = value;
    p = p+n;
    for (int i=p; i>1; i>>=1)
        tree[i>>1] = tree[i] + tree[i^1];
}
 
int query(int l, int r) {
    int res = 0;
    for (l+=n, r+=n; l<r; l>>=1, r>>=1) {
        if (l&1)
            res += tree[l++];
        if (r&1)
            res += tree[--r];
    }
    return res;
}

int main() {
    return 0;
}
```
bisect example
```
def grade(score, breakpoints=[60, 70, 80, 90], grades='FDCBA'):
    i = bisect(breakpoints, score)
    return grades[i]
[grade(score) for score in [33, 99, 77, 70, 89, 90, 100]]
['F', 'A', 'C', 'C', 'B', 'A', 'A']
```
C++ lower_bound example
```
  std::sort (v.begin(), v.end());                // 10 10 10 20 20 20 30 30

  std::vector<int>::iterator low,up;
  low=std::lower_bound (v.begin(), v.end(), 20); //          ^
  up= std::upper_bound (v.begin(), v.end(), 20); //                   ^
```
C++ set lower_bound example
```
  for (int i=1; i<10; i++) myset.insert(i*10); // 10 20 30 40 50 60 70 80 90

  itlow=myset.lower_bound (30);                //       ^
  itup=myset.upper_bound (60);                 //                   ^

  myset.erase(itlow,itup);                     // 10 20 70 80 90
```
