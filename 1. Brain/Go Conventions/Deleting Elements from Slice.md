```
*l = append(ls[:i-1], ls[i:]...)
	// *l = slices.Delete(ls, i-1, i)

```

There are couple of ways elements can be deleted from a slice. The common approach of using builtin append function. The concept is to split the main slice into two slices removing element in the middle and join them again without the desired element.

```Go
ls := []int8{2, 3, 4, 1, 10, 12, 22, 11}
*l = append(ls[:i-1], ls[i:]...)
```