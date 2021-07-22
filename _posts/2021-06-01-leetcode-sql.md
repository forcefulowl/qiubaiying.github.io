---
layout:     post
title:      leetcode sql
subtitle:   不适合阅读的整理的一些个人常用的 sql 指令
date:       2021-06-01
author:     gavin
header-img: img/post-bg-re-vs-ng2.jpg
catalog: true
tags:
    - 数据库
---

>随便整理的一些自用的sql指令

### 找中位数569

```
select Id,Company,Salary from 
(
select Id,Company,Salary,
row_number() over(partition by Company order by Salary) as rnk,
count(Salary) over(partition by Company) as cnt from Employee 
) t 
where rnk in (cnt/2,cnt/2+1,cnt/2+0.5)
```


### 找连续区间1285

```
SELECT
    MIN(log_id) as START_ID,
    MAX(log_id) as END_ID
FROM
    (SELECT
        log_id, 
        log_id - row_number() OVER(ORDER BY log_id) as num
    FROM Logs) t
GROUP BY num
```


### 1795

