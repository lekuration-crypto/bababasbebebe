import numpy as np
import matplotlib.pyplot as plt
import time

# Параметры ПСП
code_length = 511
chip_rate = 511e3
T_prn = 1e-3

SNR = 20        # в дБ

# Генерация ПСП сигнала l1of
def generate_l1of():
    # Регистр: [1, 2, 3, 4, 5, 6, 7, 8, 9]
    # 1 - вход (x⁹), 9 - выход (младший бит) ПСП снимается с 7-й ячейки
    # Начальное состояние: все единицы
    register = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    
    prn_code = np.zeros(code_length)
    
    for i in range(code_length):
        # ПСП снимается с 7-й ячейки (индекс 6 в Python)
        prn_code[i] = register[6]  # 7-я ячейка
        
        # Обратная связь: x⁵ + x⁹
        feedback = register[8] ^ register[4]  # 5-я ячейка XOR 1-я ячейка
        
        # СДВИГ ВПРАВО
        register = [feedback] + register[0:8]
    
    return prn_code

# Генерация шума 
def generate_awgn(snr_db, signal_power=1.0, signal_length=511):
    # Перевод SNR из дБ в линейный масштаб
    snr_linear = 10 ** (snr_db / 10)
    
    # Расчет мощности шума
    noise_power = signal_power / snr_linear
    
    # Генерация гауссовского шума нужной длины
    noise = np.random.normal(0, np.sqrt(noise_power), signal_length)
    
    return noise

def add_noise_to_signal(signal, snr_db):
    # Расчет мощности сигнала
    signal_power = np.mean(signal ** 2)
    
    # Генерация шума правильной длины
    noise = generate_awgn(snr_db, signal_power, len(signal))
    
    # Добавление шума к сигналу
    noisy_signal = signal + noise
    
    return noisy_signal, noise

# Для удобной записи массива в строку
def output_of_elements(signal, num_chips):
    result = ""
    for i in range(num_chips):
        chip_value = signal[i]
        chip_int = int(chip_value)
        chip_str = str(chip_int)
        result = result + chip_str
    return result

print("Генерация ПСП L1OF ГЛОНАСС с шумом SNR = 10 дБ")
print("=" * 50)

# Генерируем ПСП
l1of_code = generate_l1of()

# Проверяем начальную последовательность
first_9_chips = output_of_elements(l1of_code, 9)
print(f"Первые 9 чипов ПСП: {first_9_chips}")
print(f"Ожидается: 111111100")

# Случайные задержка и доплер
random_delay = np.random.randint(0, code_length)  # 0-510 отсчетов
random_doppler = np.random.randint(-5000, 5000)   # -5кГц до +5кГц

print(f"\nСлучайные параметры:")
print(f"Задержка: {random_delay} отсчетов")
print(f"Доплер: {random_doppler} Гц")

# Увеличиваем время накопления в 5-10 раз, чтобы нормально находить задержку по Доплеру (в ином случае возникают сильные ошибки, тк 1/1мс=1 кГц)
accumulation_factor = 10
print(f"Время накопления увеличено в {accumulation_factor} раз")

# Создаем расширенный ПСП (повторяем несколько периодов)
l1of_code_extended = np.tile(l1of_code, accumulation_factor)
extended_length = len(l1of_code_extended)

print(f"Новая длина сигнала: {extended_length} отсчетов")
print(f"Время накопления: {extended_length/chip_rate*1000:.1f} мс")

# ОБНОВЛЯЕМ сигнал с задержкой и доплером на расширенной длительности
t_extended = np.arange(extended_length) / chip_rate
phase_shift_extended = 2 * np.pi * random_doppler * t_extended

# 1. Чистый сигнал с задержкой и доплером (расширенный)
delayed_clean_signal_extended = np.roll(l1of_code_extended, random_delay) * np.exp(1j * phase_shift_extended)

# 2. Добавляем шум к расширенному сигналу
noisy_delayed_signal_extended, noise_final_extended = add_noise_to_signal(np.real(delayed_clean_signal_extended), SNR)

print("Расширенный сигнал с задержкой и доплером создан")

print(f"\nРезультаты для SNR = 10 дБ:")
print(f"Мощность сигнала: {np.mean(l1of_code**2):.3f}")
print(f"Мощность шума: {np.mean(noise_final_extended**2):.6f}")

print(f"\nСтатистики шума:")
print(f"Мат. ожидание: {np.mean(noise_final_extended):.6f}")
print(f"Дисперсия: {np.var(noise_final_extended):.6f}")
print(f"Среднеквадратичное отклонение: {np.std(noise_final_extended):.6f}")

# Статистика ПСП
ones = np.sum(l1of_code)
zeros = len(l1of_code) - ones
print(f"\nСтатистика ПСП:")
print(f"Единиц: {int(ones)} ({ones/code_length*100:.1f}%)")
print(f"Нулей: {int(zeros)} ({zeros/code_length*100:.1f}%)")

# Создаем график
plt.figure(figsize=(12, 6))

# Верхний график - сравнение сигналов
plt.subplot(2, 1, 1)
plt.plot(l1of_code[:30], 'bo-', linewidth=2, markersize=4, label='Чистый сигнал')
plt.plot(noisy_delayed_signal_extended[:30], 'ro-', linewidth=1, markersize=2, alpha=0.7, label='Сигнал + задержка + доплер + шум')
plt.title('Сравнение чистого сигнала и сигнала с задержкой, доплером и шумом')
plt.ylabel('Амплитуда')
plt.legend()
plt.grid(True)

# Нижний график - шум 
plt.subplot(2, 1, 2)
noise_component = noisy_delayed_signal_extended[:30] - np.real(delayed_clean_signal_extended[:30])
plt.plot(noise_component, 'g-', linewidth=1, label='Шум')
plt.title('Шум')
plt.xlabel('Отсчеты')
plt.ylabel('Амплитуда')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()





# Параметры поиска
chip_duration = 1 / chip_rate  # длительность одного чипа ~1.96 мкс
delta_T_samples = 1  # Шаг по задержке 1 чип
delta_F = 100  # Шаг по частоте 100 Гц

f_range = 5000  # ±5 кГц диапазон доплера
t_range_samples = code_length  # весь диапазон задержек (0-510 отсчетов)

# Расчет количества точек поиска
num_freq_points = int(2 * f_range / delta_F) + 1
num_time_points = code_length + 1

print(f"\nПараметры поиска:")
print(f"Длительность чипа: {chip_duration*1e6:.2f} мкс")
print(f"Шаг по задержке: {delta_T_samples} отсчетов ({chip_duration*1e6:.2f} мкс)")
print(f"Шаг по частоте: {delta_F} Гц")
print(f"Диапазон доплера: ±{f_range} Гц")
print(f"Диапазон задержек: 0-{t_range_samples} отсч.")
print(f"Точки поиска: {num_freq_points} по частоте, {num_time_points} по времени")
print(f"Общее количество гипотез: {num_freq_points * num_time_points}")




# ПОСЛЕДОВАТЕЛЬНЫЙ ПОИСК

# Запускаем последовательный поиск
print("\n" + "="*60)
print("ПОСЛЕДОВАТЕЛЬНЫЙ ПОИСК")
print("="*60)
def sequential_search_optimized(received_signal, reference_code):
    """
    Последовательный поиск
    """
    # Преобразуем код в формат ±1 для корреляции
    prn_code = 2 * reference_code - 1
    
    # Массивы для хранения результатов
    correlation_grid = np.zeros((num_freq_points, num_time_points))
    freq_grid = np.zeros(num_freq_points)
    delay_grid = np.zeros(num_time_points)
    
    max_correlation = 0
    best_doppler = 0
    best_delay = 0
    
    print("\nНачало последовательного поиска")
    start_time = time.time()
    search_count = 0
    
    # Перебор по частоте Доплера
    for i in range(num_freq_points):
        current_freq = -f_range + i * delta_F
        freq_grid[i] = current_freq
        
        # Перебор по задержке
        for j in range(num_time_points):
            current_delay = j * delta_T_samples
            delay_grid[j] = current_delay
            
            # Компенсация задержки и доплера в принятом сигнале
            t = np.arange(len(received_signal)) / chip_rate
            
            # 1. Компенсируем доплер
            doppler_compensated = received_signal * np.exp(-1j * 2 * np.pi * current_freq * t)
            
            # 2. Компенсируем задержку 
            fully_compensated = np.roll(doppler_compensated, -current_delay)
            
            # Вычисляем корреляцию с опорным кодом
            correlation = np.abs(np.dot(fully_compensated, prn_code))
            correlation_grid[i, j] = correlation
            
            search_count += 1
            
            # Обновляем максимум
            if correlation > max_correlation:
                max_correlation = correlation
                best_doppler = current_freq
                best_delay = current_delay
    
    search_time = time.time() - start_time
    
    print(f"\nРезультаты последовательного поиска:")
    print(f"Всего проверено гипотез: {search_count}")
    print(f"Время поиска: {search_time:.3f} сек")
    print(f"Найденная задержка: {best_delay} отсчетов ({best_delay/chip_rate*1e6:.2f} мкс)")
    print(f"Найденный доплер: {best_doppler} Гц")
    print(f"Максимальная корреляция: {max_correlation:.2f}")
    return best_delay, best_doppler, max_correlation, freq_grid, delay_grid, correlation_grid

# Преобразуем сигнал в комплексный формат для поиска
complex_received_signal = noisy_delayed_signal_extended.astype(complex)

# Запускаем последовательный поиск
found_delay_opt, found_doppler_opt, max_corr_opt, freq_grid_opt, delay_grid_opt, corr_grid_opt = sequential_search_optimized(
    complex_received_signal, l1of_code_extended 
)



# Визуализация для последовательного поиска

# Окно 1: 2D карта корреляции
plt.figure(figsize=(10, 6))
plt.contourf(delay_grid_opt, freq_grid_opt / 1000, corr_grid_opt, levels=30, cmap='viridis')
plt.colorbar(label='Корреляция')
plt.xlabel('Задержка (отсчеты)')
plt.ylabel('Доплер (кГц)')
plt.title('2D карта корреляции')
plt.axvline(x=random_delay, color='red', linestyle='--', label='Истинная задержка')
plt.axhline(y=random_doppler/1000, color='red', linestyle='--', label='Истинный доплер')
plt.scatter(found_delay_opt, found_doppler_opt/1000, color='yellow', s=100, marker='*', label='Найденный максимум')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()



# Окно 2: 3D поверхность корреляции
fig = plt.figure(figsize=(10, 7))
ax_3d = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(delay_grid_opt, freq_grid_opt / 1000)
surf = ax_3d.plot_surface(X, Y, corr_grid_opt, cmap='viridis', alpha=0.8)
ax_3d.scatter(found_delay_opt, found_doppler_opt/1000, max_corr_opt, 
             color='red', s=50, marker='o', label='Вершина')
ax_3d.set_xlabel('Задержка (отсчеты)')
ax_3d.set_ylabel('Доплер (кГц)')
ax_3d.set_zlabel('Корреляция')
ax_3d.view_init(elev=20, azim=45)
fig.colorbar(surf, label='Корреляция')
plt.tight_layout()
plt.show()



# Проверка точности
delay_error_samples = abs(found_delay_opt - random_delay)
doppler_error_hz = abs(found_doppler_opt - random_doppler)

delay_accuracy = "ВЫСОКАЯ точность" if delay_error_samples <= delta_T_samples else "СРЕДНЯЯ точность" if delay_error_samples <= 2*delta_T_samples else "НИЗКАЯ точность"
doppler_accuracy = "ВЫСОКАЯ точность" if doppler_error_hz <= delta_F else "СРЕДНЯЯ точность" if doppler_error_hz <= 2*delta_F else "НИЗКАЯ точность"

print(f"\nТочность последовательного поиска:")
print(f"Задержка: {delay_accuracy} (ошибка {delay_error_samples} отсч. при шаге {delta_T_samples} отсч.)")
print(f"Доплер: {doppler_accuracy} (ошибка {doppler_error_hz} Гц при шаге {delta_F} Гц)")





# ПАРАЛЛЕЛЬНЫЙ ПОИСК

def parallel_correlators_search(received_signal, reference_code, num_correlators=16):
    """
    Параллельный поиск с использованием виртуальных корреляторов
    """
    # Используем глобальную переменную code_length
    global code_length
    
    # Преобразуем код в формат ±1 для корреляции
    prn_code = 2 * reference_code - 1
    
    # Массивы для хранения результатов
    correlation_grid = np.zeros((num_freq_points, num_time_points))
    freq_grid = np.zeros(num_freq_points)
    delay_grid = np.zeros(num_time_points)
    
    max_correlation = 0
    best_doppler = 0
    best_delay = 0
    
    print(f"\nНачало параллельного поиска ({num_correlators} корреляторов)")
    start_time = time.time()
    search_count = 0
    
    # Перебор по частоте Доплера (последовательный)
    for i in range(num_freq_points):
        current_freq = -f_range + i * delta_F
        freq_grid[i] = current_freq
        
        # Компенсация доплера для всей частоты
        t = np.arange(len(received_signal)) / chip_rate
        compensated_signal = received_signal * np.exp(-1j * 2 * np.pi * current_freq * t)
        
        # ПАРАЛЛЕЛЬНАЯ обработка задержек блоками
        delays_per_block = num_time_points // num_correlators
        
        for block_idx in range(0, num_time_points, delays_per_block):
            block_end = min(block_idx + delays_per_block, num_time_points)
            
            # Обрабатываем блок задержек
            for j in range(block_idx, block_end):
                current_delay = j * delta_T_samples
                delay_grid[j] = current_delay
                
                # Компенсация задержки
                fully_compensated = np.roll(compensated_signal, -current_delay)
                
                # Корреляция
                correlation = np.abs(np.dot(fully_compensated, prn_code))
                correlation_grid[i, j] = correlation
                search_count += 1
                
                # Обновляем максимум
                if correlation > max_correlation:
                    max_correlation = correlation
                    best_doppler = current_freq
                    best_delay = current_delay
    
    search_time = time.time() - start_time
    
    print(f"\nРезультаты параллельного поиска:")
    print(f"Всего проверено гипотез: {search_count}")
    print(f"Время поиска: {search_time:.3f} сек")
    print(f"Найденная задержка: {best_delay} отсчетов")
    print(f"Найденный доплер: {best_doppler} Гц")
    print(f"Максимальная корреляция: {max_correlation:.2f}")
    
    return best_delay, best_doppler, max_correlation, freq_grid, delay_grid, correlation_grid

# Запускаем параллельный поиск
print("\n" + "="*60)
print("ПАРАЛЛЕЛЬНЫЙ ПОИСК")
print("="*60)

found_delay_parallel, found_doppler_parallel, max_corr_parallel, freq_grid_parallel, delay_grid_parallel, corr_grid_parallel = parallel_correlators_search(
    complex_received_signal, l1of_code_extended, num_correlators=16
)



# Визуализация для параллельного поиска

# Окно 1: 2D карта корреляции
plt.figure(figsize=(10, 6))
plt.contourf(delay_grid_opt, freq_grid_opt / 1000, corr_grid_opt, levels=30, cmap='viridis')
plt.colorbar(label='Корреляция')
plt.xlabel('Задержка (отсчеты)')
plt.ylabel('Доплер (кГц)')
plt.title('2D карта корреляции')
plt.axvline(x=random_delay, color='red', linestyle='--', label='Истинная задержка')
plt.axhline(y=random_doppler/1000, color='red', linestyle='--', label='Истинный доплер')
plt.scatter(found_delay_opt, found_doppler_opt/1000, color='yellow', s=100, marker='*', label='Найденный максимум')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()



# 3D визуализация для параллельного поиска
fig = plt.figure(figsize=(10, 7))
ax_3d = fig.add_subplot(111, projection='3d')
X_parallel, Y_parallel = np.meshgrid(delay_grid_parallel, freq_grid_parallel / 1000)
surf_parallel = ax_3d.plot_surface(X_parallel, Y_parallel, corr_grid_parallel, cmap='viridis', alpha=0.8)
ax_3d.scatter(found_delay_parallel, found_doppler_parallel/1000, max_corr_parallel, 
             color='red', s=50, marker='o', label='Вершина')
ax_3d.set_xlabel('Задержка (отсчеты)')
ax_3d.set_ylabel('Доплер (кГц)')
ax_3d.set_zlabel('Корреляция')
ax_3d.view_init(elev=20, azim=45)
fig.colorbar(surf_parallel, label='Корреляция')
plt.tight_layout()
plt.show()



# Проверка точности для параллельного поиска
delay_error_parallel = abs(found_delay_parallel - random_delay)
doppler_error_parallel = abs(found_doppler_parallel - random_doppler)

print(f"\nТочность параллельного поиска:")
print(f"Задержка: {delay_accuracy} (ошибка {delay_error_parallel} отсч.)")
print(f"Доплер: {doppler_accuracy} (ошибка {doppler_error_parallel} Гц)")



# # Сравнение результатов
# print("\n" + "="*60)
# print("СРАВНЕНИЕ МЕТОДОВ ПОИСКА")
# print("="*60)
# print(f"Истинные параметры: задержка={random_delay} отсч., доплер={random_doppler} Гц")

# print(f"\nПОСЛЕДОВАТЕЛЬНЫЙ ПОИСК:")
# print(f"Задержка: {found_delay_opt} отсч. (ошибка: {abs(found_delay_opt - random_delay)} отсч.)")
# print(f"Доплер: {found_doppler_opt} Гц (ошибка: {abs(found_doppler_opt - random_doppler)} Гц)")

# print(f"\nПАРАЛЛЕЛЬНЫЙ ПОИСК:")
# print(f"Задержка: {found_delay_parallel} отсч. (ошибка: {abs(found_delay_parallel - random_delay)} отсч.)")
# print(f"Доплер: {found_doppler_parallel} Гц (ошибка: {abs(found_doppler_parallel - random_doppler)} Гц)")
