* To execute the program, you can run this command 

```shell
./train.py 
```

* by default, it runs all the available algorithm where each with 3 epoch and in each epoch, it runs 150 iterations

* ```shell
  --epoch #control the number of epoch
  --hidden_dim #the number of components in the dictionary
  --iters #the number of iterations
  --tol #the threshold for early stopping
  
  --NMF_OBJ #the algorithm, when by default is None, can be either {L1, L21, L2}
  
  --root #the root of data
  --reduce #to reduce the size of image, expecting int
  --split_ratio #split data into training and testing, expecting float
  
  --noise # types of noise, by dafault is gaussian noise, can also be {salt_and_pepper}
  --p # to control the percentage of contaminated pixel point,  for salt_and_pepper
  --r # to control the percentage of while point in the salt_and_pepper noise
  --mu #the mean value for gaussian noise
  --sigma # the standard deviation for the gaussian noise
  
  --save_rres #by default False, save all the rre in each iters, save to /result
  
  --save_np # save the matrix of dictionary D and representation R as well as an auxiliary matrix E, save to npys, by default False
  ```

* 