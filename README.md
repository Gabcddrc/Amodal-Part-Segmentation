# PROJECT 5 - AMODAL PART SEGMENTATION
## Team MP123
### Team member
- Zhenjie Jiang     
zhejiang@ethz.ch
-  Shuaijun Gao <br />
shugao@ethz.ch
-  Xintian Yuan <br />
xinyuan@sethz.ch


## Re-produce leaderboard result
```console
cd mp_segm
python train.py --batch_size 32  --trainsplit trainval
python test.py --load_ckpt logs/$EXP_KEY/latest.pt
```
requirements.txt contains all the libraries needed