#!/bin/bash

N=15000
Ts=(1 2 4 8 16 24)
programs=("bin/omp1" "bin/omp2")
output_file="measurements"

# Проверка наличия программ
for prog in "${programs[@]}"; do
	if [ ! -f "$prog" ]; then
    	echo "Ошибка: программа $prog не найдена"
    	exit 1
	fi
done

# Очистка файла результатов
> "$output_file"

# Запуск тестов для каждой программы
for prog in "${programs[@]}"; do
	results=()
	echo "Тестируем $prog..."

	for T in "${Ts[@]}"; do
    	min_time=999999
    	echo "Тестируем T=$T..."

    	# 5 запусков для каждого T
    	for i in {1..5}; do
        	output=$($prog $N $T 2>&1)
        	time=$(echo "$output" | grep -oP 'Time: \K\d+\.\d+')
        	echo "$i : $time"

        	# Обновление минимального времени
        	if (( $(echo "$time < $min_time" | bc -l) )); then
            	min_time=$time
        	fi
    	done

    	results+=("$min_time")
	done

	# Форматирование результатов в строку
	echo "${results[@]}" | tr ' ' '\t' >> "$output_file"
done

echo "Результаты сохранены в $output_file"
