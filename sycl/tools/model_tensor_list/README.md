## Run

feng-
ghp
_zKN
qBwfuhh0
ZrwyUGH7lzgzhkNI3st32nKuW

Huggingface token
d9801
h
f
_
QQJzVj
iLQrnOowKTDBJtzGweVwCefPCbYw

cding-nv
ding9@163.com 00!

鼠标滚轴 变 history 问题， 用下面的方法
export TERM=xterm-256color  
source .bashrc
tmux new -s itex
tmux ls
tmux attach -t 0
tmux a -t 0

tmux attach -t <session-name>
tmux kill-session -t 0 or session-name
tmux switch -t 0 切换
tmux rename-session -t 0<new-name>

Tmux info
 tmux kill-session -t 0

tmux kill-server  关闭所有


etrade
53383885 - INTC
ding9801_intel
1F


Form W-8BEN is submitted by a non-resident of the United States to establish foreign status and to claim tax treaty benefits.
Submit a Form W-8BEN or you may be taxed up to 30%, the default rate.

https://us.etrade.com/forms-applications   -> W-8BEN Certificate of Foreign Status

```
Vscode 免密配置
C:\Users\fengding\.ssh  把 id_rsa.pub 拷贝的 server
在 server 端     
   $ mkdir .ssh
   $ cd .ssh
   $ cat  ../id_rsa.pub > ./authorized_keys    
    $ service  sshd restart
	
修改 C:\Users\fengding\.ssh    config，  加上   IdentityFile C:\Users\fengding\.ssh\id_rsa   即可
```
```
https://www.wsj.com/tech/ai
https://www.reuters.com/technology/            https://www.reuters.com/technology/artificial-intelligence/
https://techxplore.com/machine-learning-ai-news/
https://cybernews.com/tech/
https://techcrunch.com/category/artificial-intelligence/
https://siliconangle.com/category/ai/
https://www.theinformation.com/technology
https://www.nist.gov/news-events/news/search?key=artifical+intelligence+&topic-op=or  
https://www.msn.com/en-us/money/markets
https://www.nature.com/news
https://www.cgchannel.com/category/news/
https://www.theverge.com/tech
https://techstartups.com/
https://www.theverge.com/ai-artificial-intelligence

```

```
Start a new virtual env
rm ~/.venv/lint 
python3 -m venv ~/.venv/lint 
source ~/.venv/lint/bin/activate


$ git pull 

for example PR 458, 
$ git fetch origin pull/458/head:Your_branch_name_PR_458
The new branch Your_branch_name_PR_458 is what you want

```
```
吴恩达课程
    https://github.com/HeartyHaven/prompt-engineering-for-developers  
    https://learn.deeplearning.ai/
     https://www.bilibili.com/video/BV1Bo4y1A7FU/?share_source=copy_web  
    https://github.com/datawhalechina/llm-cookbook
    https://www.bilibili.com/video/BV11G411X7nZ/
```



```
$ python convert_awq_to_bin.py your_model_bin 
$ python  saftetensors_load.py your_model_bin
```
