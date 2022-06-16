num_preson=int(input())
start=list(map(int,input.split()))
end=list(map(int,input().split()))
def main(n, start, end):
    result=0
    end1,start1 = (list(t)for t in zip(*sorted(zip(end,start))))
    start_time=1
    for i in range(n):
        if start1[i]>=start_time:
            result+=1
            start_time=end1[ i]+1
        else:
            continue
    return result
res=main(num_preson, start, end)
print(res)
