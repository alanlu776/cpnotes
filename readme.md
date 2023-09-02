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

## Data Structures
### Priority queue
`heapq.heappush(heap, item)`
Push the value item onto the heap, maintaining the heap invariant.

`heapq.heappop(heap)`
Pop and return the smallest item from the heap, maintaining the heap invariant. If the heap is empty, IndexError is raised. To access the smallest item without popping it, use heap[0].

`heapq.heappushpop(heap, item)`
Push item on the heap, then pop and return the smallest item from the heap. The combined action runs more efficiently than heappush() followed by a separate call to heappop().

`heapq.heapify(x)`
Transform list x into a heap, in-place, in linear time.

`heapq.heapreplace(heap, item)`
Pop and return the smallest item from the heap, and also push the new item. The heap size doesn’t change. If the heap is empty, IndexError is raised.

This one step operation is more efficient than a heappop() followed by heappush() and can be more appropriate when using a fixed-size heap. The pop/push combination always returns an element from the heap and replaces it with item.

The value returned may be larger than the item added. If that isn’t desired, consider using heappushpop() instead. Its push/pop combination returns the smaller of the two values, leaving the larger value on the heap.

### Multiset
Use Python `collections.Counter(iterable/mapping)` instead. 

A Counter is a dict subclass for counting hashable objects. It is a collection where elements are stored as dictionary keys and their counts are stored as dictionary values. Counts are allowed to be any integer value including zero or negative counts. The Counter class is similar to bags or multisets in other languages.

Elements are counted from an iterable or initialized from another mapping (or counter):

```
>>>
c = Counter()                           # a new, empty counter
c = Counter('gallahad')                 # a new counter from an iterable
c = Counter({'red': 4, 'blue': 2})      # a new counter from a mapping
c = Counter(cats=4, dogs=8)             # a new counter from keyword args
```
Counter objects have a dictionary interface except that they return a zero count for missing items instead of raising a KeyError:

```
>>>
c = Counter(['eggs', 'ham'])
c['bacon']                              # count of a missing element is zero
0
```
Setting a count to zero does not remove an element from a counter. Use del to remove it entirely:

```
>>>
c['sausage'] = 0                        # counter entry with a zero count
del c['sausage']                        # del actually removes the entry
```
New in version 3.1.

Changed in version 3.7: As a dict subclass, Counter inherited the capability to remember insertion order. Math operations on Counter objects also preserve order. Results are ordered according to when an element is first encountered in the left operand and then by the order encountered in the right operand.

Counter objects support additional methods beyond those available for all dictionaries:

`elements()`
Return an iterator over elements repeating each as many times as its count. Elements are returned in the order first encountered. If an element’s count is less than one, elements() will ignore it.

```
>>>
c = Counter(a=4, b=2, c=0, d=-2)
sorted(c.elements())
['a', 'a', 'a', 'a', 'b', 'b']
```
`most_common([n])`
Return a list of the n most common elements and their counts from the most common to the least. If n is omitted or None, most_common() returns all elements in the counter. Elements with equal counts are ordered in the order first encountered:

```
>>>
Counter('abracadabra').most_common(3)
[('a', 5), ('b', 2), ('r', 2)]
```
`subtract([iterable-or-mapping])`
Elements are subtracted from an iterable or from another mapping (or counter). Like dict.update() but subtracts counts instead of replacing them. Both inputs and outputs may be zero or negative.

```
>>>
c = Counter(a=4, b=2, c=0, d=-2)
d = Counter(a=1, b=2, c=3, d=4)
c.subtract(d)
c
Counter({'a': 3, 'b': 0, 'c': -3, 'd': -6})
```
New in version 3.2.

`total()`
Compute the sum of the counts.

```
>>>
c = Counter(a=10, b=5, c=0)
c.total()
15
```

New in version 3.10.

The usual dictionary methods are available for Counter objects except for two which work differently for counters.

`fromkeys(iterable)`
This class method is not implemented for Counter objects.

`update([iterable-or-mapping])`
Elements are counted from an iterable or added-in from another mapping (or counter). Like dict.update() but adds counts instead of replacing them. Also, the iterable is expected to be a sequence of elements, not a sequence of (key, value) pairs.

Counters support rich comparison operators for equality, subset, and superset relationships: ==, !=, <, <=, >, >=. All of those tests treat missing elements as having zero counts so that Counter(a=1) == Counter(a=1, b=0) returns true.

New in version 3.10: Rich comparison operations were added.

Changed in version 3.10: In equality tests, missing elements are treated as having zero counts. Formerly, Counter(a=3) and Counter(a=3, b=0) were considered distinct.

Common patterns for working with Counter objects:

```
c.total()                       # total of all counts
c.clear()                       # reset all counts
list(c)                         # list unique elements
set(c)                          # convert to a set
dict(c)                         # convert to a regular dictionary
c.items()                       # convert to a list of (elem, cnt) pairs
Counter(dict(list_of_pairs))    # convert from a list of (elem, cnt) pairs
c.most_common()[:-n-1:-1]       # n least common elements
+c                              # remove zero and negative counts
```

Several mathematical operations are provided for combining Counter objects to produce multisets (counters that have counts greater than zero). Addition and subtraction combine counters by adding or subtracting the counts of corresponding elements. Intersection and union return the minimum and maximum of corresponding counts. Equality and inclusion compare corresponding counts. Each operation can accept inputs with signed counts, but the output will exclude results with counts of zero or less.

```
>>>
c = Counter(a=3, b=1)
d = Counter(a=1, b=2)
c + d                       # add two counters together:  c[x] + d[x]
Counter({'a': 4, 'b': 3})
c - d                       # subtract (keeping only positive counts)
Counter({'a': 2})
c & d                       # intersection:  min(c[x], d[x])
Counter({'a': 1, 'b': 1})
c | d                       # union:  max(c[x], d[x])
Counter({'a': 3, 'b': 2})
c == d                      # equality:  c[x] == d[x]
False
c <= d                      # inclusion:  c[x] <= d[x]
False
Unary addition and subtraction are shortcuts for adding an empty counter or subtracting from an empty counter.

>>>
c = Counter(a=2, b=-4)
+c
Counter({'a': 2})
-c
Counter({'b': 4})
```

### Balanced BST
No implementation in Python. Use `bisect`.

This module provides support for maintaining a list in sorted order without having to sort the list after each insertion. For long lists of items with expensive comparison operations, this can be an improvement over the more common approach. The module is called bisect because it uses a basic bisection algorithm to do its work. The source code may be most useful as a working example of the algorithm (the boundary conditions are already right!).

The following functions are provided:

`bisect.bisect_left(a, x, lo=0, hi=len(a), *, key=None)`
Locate the insertion point for x in a to maintain sorted order. The parameters lo and hi may be used to specify a subset of the list which should be considered; by default the entire list is used. If x is already present in a, the insertion point will be before (to the left of) any existing entries. The return value is suitable for use as the first parameter to list.insert() assuming that a is already sorted.

The returned insertion point i partitions the array a into two halves so that all(val < x for val in a[lo : i]) for the left side and all(val >= x for val in a[i : hi]) for the right side.

key specifies a key function of one argument that is used to extract a comparison key from each element in the array. To support searching complex records, the key function is not applied to the x value.

If key is None, the elements are compared directly with no intervening function call.

Changed in version 3.10: Added the key parameter.

`bisect.bisect_right(a, x, lo=0, hi=len(a), *, key=None)`
`bisect.bisect(a, x, lo=0, hi=len(a), *, key=None)`
Similar to bisect_left(), but returns an insertion point which comes after (to the right of) any existing entries of x in a.

The returned insertion point i partitions the array a into two halves so that all(val <= x for val in a[lo : i]) for the left side and all(val > x for val in a[i : hi]) for the right side.

key specifies a key function of one argument that is used to extract a comparison key from each element in the array. To support searching complex records, the key function is not applied to the x value.

If key is None, the elements are compared directly with no intervening function call.

Changed in version 3.10: Added the key parameter.

`bisect.insort_left(a, x, lo=0, hi=len(a), *, key=None)`
Insert x in a in sorted order.

This function first runs bisect_left() to locate an insertion point. Next, it runs the insert() method on a to insert x at the appropriate position to maintain sort order.

To support inserting records in a table, the key function (if any) is applied to x for the search step but not for the insertion step.

Keep in mind that the O(log n) search is dominated by the slow O(n) insertion step.

Changed in version 3.10: Added the key parameter.

`bisect.insort_right(a, x, lo=0, hi=len(a), *, key=None)`
`bisect.insort(a, x, lo=0, hi=len(a), *, key=None)`
Similar to insort_left(), but inserting x in a after any existing entries of x.

This function first runs bisect_right() to locate an insertion point. Next, it runs the insert() method on a to insert x at the appropriate position to maintain sort order.

To support inserting records in a table, the key function (if any) is applied to x for the search step but not for the insertion step.

Keep in mind that the O(log n) search is dominated by the slow O(n) insertion step.

Changed in version 3.10: Added the key parameter.

### Fenwick tree
Used for dynamic cumulative frequency table, for range sums with many updates.

```python
class FTree:
    def __init__(self, f):
        self.n = len(f)
        self.ft = [0] * (self.n + 1)

        for i in range(1, self.n + 1):
            self.ft[i] += f[i - 1]
            if i + self.lsone(i) <= self.n:
                self.ft[i + self.lsone(i)] += self.ft[i]

    def lsone(self, s):
        return s & (-s)

    # Range sum query, the number of elements between i and j, including i, excluding j
    def query(self, i, j):
        if i > 1:
            return self.query(1, j) - self.query(1, i - 1)

        s = 0
        while j > 0:
            s += self.ft[j]
            j -= self.lsone(j)

        return s

    # Add v to element/index i
    def update(self, i, v):
        while i <= self.n:
            self.ft[i] += v
            i += self.lsone(i)

    # Order statistics, smallest index/key i so that the cumulative frequency [1..i] using 1-indexing >= k. Does BSearch.
    def select(self, k):
        p = 1
        while (p * 2) <= self.n: p *= 2

        i = 0
        while p > 0:
            if k > self.ft[i + p]:
                k -= self.ft[i + p]
                i += p
            p //= 2

        return i + 1

class RUPQ:
    def __init__(self, n):
        self.ftree = FTree([0] * n)

    def query(self, i):
        return self.ftree.query(1, i)

    # Update range inclusive from i to j, +v. Keeps log m single query.
    def update(self, i, j, v):
        self.ftree.update(i, v)
        self.ftree.update(j + 1, -v)

class RURQ:
    def __init__(self, n):
        self.f = FTree([0] * n)
        self.r = RUPQ(n)

    # Keeps efficient query
    def query(self, i, j):
        if i > 1:
            return self.query(1, j) - self.query(1, i - 1)
        return self.r.query(j) * j - self.f.query(1, j)

    def update(self, i, j, v):
        self.r.update(i, j, v)
        self.f.update(i, v * (i - 1))
        self.f.update(j + 1, -1 * v * j)


f = [0, 1, 0, 1, 2, 3, 2, 1, 1, 0]
ft = FTree(f)
print(ft.query(1, 6) == 7)
print(ft.query(1, 3) == 1)
print(ft.select(7) == 6)
ft.update(5, 1)
print(ft.query(1, 10) == 12)

r = RUPQ(10)
r.update(2, 9, 7)
r.update(6, 7, 3)
print(r.query(1) == 0)
print(r.query(2) == 7)
print(r.query(3) == 7)
print(r.query(4) == 7)
print(r.query(5) == 7)
print(r.query(6) == 10)
print(r.query(7) == 10)
print(r.query(8) == 7)
print(r.query(9) == 7)
print(r.query(10) == 0)

r = RURQ(10)
r.update(2, 9, 7)
r.update(6, 7, 3)
print(r.query(3, 5) == 21)
print(r.query(7, 8) == 17)
```

