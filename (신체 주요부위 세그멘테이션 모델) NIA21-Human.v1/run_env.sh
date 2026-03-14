
# --------------
# 시험환경
# 각 항목별로 스샷과 로그 제출

# --------------
# CPU
date
lscpu > log/01.CPU.txt
lscpu

# cat /proc/cpuinfo
# sudo dmidecode -t processor

# --------------
# GPU
date
nvidia-smi > log/02.GPU.txt
nvidia-smi

# --------------
# RAM
date
sudo dmidecode -t memory > log/03.RAM.txt
sudo dmidecode -t memory

# --------------
# HDD
date
df -h > log/04.HDD.txt
df -h

# --------------
# OS
date
uname -a > log/05.OS.txt
uname -a

# --------------
# Framework
date
python -V > log/06.Framework.txt
python -V
pip list >> log/06.Framework.txt
pip list




# --------------
# 시험수행
# 시험 시작, 중간, 끝 시점에 각각 스샷 제출
# output 폴더의 log, csv 파일 제출
date
cat docker-compose.yml
ls -l output
docker-compose up
ls -l output
date
