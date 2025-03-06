#!/bin/bash

N=20000
Ts=(1 2 4 8 16)
programs=("bin/mpi1" "bin/mpi2")
sources=("src/Cmpi1.c" "src/Cmpi2.c")
output_file="measurements"

# Проверка наличия mpicxx
if ! command -v mpicxx &> /dev/null; then
    echo "Ошибка: mpicxx не найден. Убедитесь, что MPI установлен."
    exit 1
fi

# Компиляция программ
echo "Компиляция программ..."
for i in "${!programs[@]}"; do
    prog=${programs[$i]}
    src=${sources[$i]}
    
    if [ ! -f "$src" ]; then
        echo "Ошибка: исходный файл $src не найден"
        exit 1
    fi
    
    echo "Компиляция $src -> $prog..."
    mpicxx -std=c++11 -O3 -Wall -Wextra "$src" -o "$prog"
    if [ $? -ne 0 ]; then
        echo "Ошибка компиляции $src"
        exit 1
    fi
done

# Проверка наличия скомпилированных программ
for prog in "${programs[@]}"; do
    if [ ! -f "$prog" ]; then
        echo "Ошибка: программа $prog не найдена после компиляции"
        exit 1
    fi
done

# Проверка наличия mpirun
if ! command -v mpirun &> /dev/null; then
    echo "Ошибка: mpirun не найден. Убедитесь, что MPI установлен."
    exit 1
fi

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
            # Запуск программы с mpirun и указанием числа процессов
            output=$(mpirun -np $T $prog $N 2>&1)
            time=$(echo "$output" | grep -oP 'time: \K\d+\.\d+')
            
            # Проверка, что время удалось извлечь
            if [ -z "$time" ]; then
                echo "$i : Не удалось извлечь время из вывода"
                time=999999  # Устанавливаем большое значение в случае ошибки
            else
                echo "$i : $time"
            fi

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