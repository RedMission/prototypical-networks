#!/usr/bin/env bash
DATADIR=data/omniglot/data

mkdir -p $DATADIR
# Windows 默认没有 wget 命令。但你已经安装了 Git，并且 Git 的 sh.exe 被添加到了系统的 PATH 变量中。你可以使用 Git 的 curl 命令代替 wget。
#wget -O images_background.zip https://github.com/brendenlake/omniglot/blob/master/python/images_background.zip?raw=true
#wget -O images_evaluation.zip https://github.com/brendenlake/omniglot/blob/master/python/images_evaluation.zip?raw=true

curl -O https://github.com/brendenlake/omniglot/blob/master/python/images_background.zip?raw=true
curl -O https://github.com/brendenlake/omniglot/blob/master/python/images_evaluation.zip?raw=true

unzip images_background.zip -d $DATADIR
unzip images_evaluation.zip -d $DATADIR
mv $DATADIR/images_background/* $DATADIR/
mv $DATADIR/images_evaluation/* $DATADIR/
rmdir $DATADIR/images_background
rmdir $DATADIR/images_evaluation
