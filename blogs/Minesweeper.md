---
title: CSC3100 Minesweeper题解
description: CSC3100 Minesweeper题解
copyright_author: Danny Chen
copyright_url: https://baichengdanny.github.io
copyright_info: 此文章版权归Danny Chen所有，如有转载，请注明来自原作者
mathjax: true
---
# Minesweeper

## Problem

### DESCRIPTION

Suppose we are playing Minesweeper in a $$2 × N$$ map, where there are some mines buried in the grids in the first row but none in the second row. If a grid have no mine, it will display the number of mines that is adjacent to it in 8 directions.

Given an array $$A$$ indicates the numbers each grids in the second row display, you need to calculate the number of all valid layout of mines in the first row.

### INPUT

The first line contains a number $$N$$, where $$1 ≤ N ≤ 10000$$. The second line contains N numbers, where the $$i-th$$ number is $$Ai$$.

### OUTPUT

Only one integer, which is the number of all valid layout of mines in the first row.

### SAMPLE INPUT

```
2
1 1
```

### SAMPLE OUTPUT

```
2
```

### SCORING

For 60% test cases, $$N ≤ 1000$$.

For 100% test cases, $$N ≤ 10000$$.

## Ideas

> 题解先用中文说明，后面会附有英文翻译结果，翻译由DeepL提供

显然，我们可以用动态规划的思想解决这个问题，从题意可知，这是一道计数类DP问题。首先，定义一个二维数组$$f[i][j]$$，表示当指针走到第$$i$$列，状态为$$j$$时的所有地雷放置方法。我们要求得的是地雷放置方法的数量。但本题的状态转移方程并不好想。可能在定义时大家已经看出来了，我们设置了一个状态$$j$$，它表示我们目前地雷的放置状态。

从状态切入，我们分四种状态$$j$$来推导状态转移方程：

1. $j = 0$，表示当前位置i的不是地雷，并且i+1也不是雷 
2. $j = 1$，表示当前位置i的不是地雷，并且i+1是地雷
3. $j = 2$，表示当前位置i的是地雷，并且i+1不是地雷 
4. $j = 3$，表示当前位置i的是地雷，并且i+1也是地雷

所以，

```
f[i][0]:表示当前位置i的不是地雷，并且i+1也不是雷 
f[i][1]:表示当前位置i的不是地雷，并且i+1是地雷 
f[i][2]:表示当前位置i的是地雷，并且i+1不是地雷 
f[i][3]:表示当前位置i的是地雷，并且i+1也是地雷 
```

有了这四种状态，我们就可以很轻松的得出状态转移方程了。具体请看下面的代码。

Obviously, we can use the idea of dynamic programming to solve this problem, which is a counting-type DP problem as we can see from the question. First, define a two-dimensional array $$f[i][j]$$ that represents all mine placement methods when the pointer goes to column $$i$$ and the state is $$j$$. What we are looking for is the number of mine placement methods. However, the state transfer equation in this problem is not well thought out. As you may have seen in the definition, we set up a state $$j$$ which represents our current state of mine placement.

Cutting from the state, we derive the state transfer equation in four states $$j$$:

1. $$j = 0$$, which means that the current position i is not a mine, and i+1 is not a mine either. 
2. $$j = 1$$, which means that the current position i is not a mine and i+1 is a mine
3. $j = 2$, means that current position i is a mine and i+1 is not a mine. 
4. $j = 3$, which means that the current position i is a mine and i+1 is also a mine.

So,

```
f[i][0]: means current position i is not a mine, and i+1 is not a mine. 
f[i][1]: means current position i is not a mine, and i+1 is a mine. 
f[i][2]: means current position i is a mine, and i+1 is not a mine. 
f[i][3]: means current position i is a mine and i+1 is also a mine 
```

With these four states, we can easily derive the state transfer equation. Please see the following code for details.

## C++ Code

> You can transfer to another language using ChatGPT.

```c++
#include <bits/stdc++.h>
using namespace std;

typedef long long ll;

const int N = 10010;

int n, ans = 0, a[N];
ll f[N][4];

int main(){
	cin >> n;
	for(int i = 1; i <= n; i ++ ) cin >> a[i];
	
    //初始化第一个位置
	if(a[1] == 0) f[1][0] = 1;
	else if(a[1] == 1) f[1][1] = f[1][2] = 1;
	else if(a[1] == 2) f[1][3] = 1;
    
    //分情况写状态转移方程
	for(int i = 2; i < n; i ++ ){
		if(a[i] == 0) //显示有0个地雷时
            f[i][0] = f[i - 1][0]; // 左、中(当前i)、右都不是雷
		if(a[i] == 1) { //显示有1个地雷时
			f[i][0] = f[i - 1][2]; // 左是雷
			f[i][1] = f[i - 1][0]; // 右是雷
			f[i][2] = f[i - 1][1]; // 中是雷
		}
		if(a[i] == 2) { //显示有2个地雷时
			f[i][2] = f[i - 1][3]; // 左、中是雷
			f[i][1] = f[i - 1][2]; // 左、右是雷
			f[i][3] = f[i - 1][1]; // 中、右是雷
		}
		if(a[i] == 3) { //显示有3个地雷时(因为是二维，最多显示有3个地雷)
			f[i][3] = f[i - 1][3]; // 左、中、右都是雷
		}
	}
	
    // 计算i走到最后统计的所有方法数
	if(a[n] == 0) ans += f[n - 1][0];
	if(a[n] == 1) ans += f[n - 1][1] + f[n - 1][2];
	if(a[n] == 2) ans += f[n - 1][3];
	
	cout << ans << endl;
	return 0;
}
```

## Similar Problem for practicing

[Problem - D - Codeforces](https://codeforces.com/contest/404/problem/D)
