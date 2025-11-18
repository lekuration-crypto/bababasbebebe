import numpy as np
import matplotlib.pyplot as plt
import time

# Параметры ПСП
code_length = 511
chip_rate = 511e3
T_prn = 1e-3

SNR = 40        # в дБ

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
def generate_awgn(snr_db, signal_power=1.0):
    # Перевод SNR из дБ в линейный масштаб
    snr_linear = 10 ** (snr_db / 10)
    
    # Расчет мощности шума
    noise_power = signal_power / snr_linear
    
    # Генерация гауссовского шума для расширенного сигнала
    noise = np.random.normal(0, np.sqrt(noise_power), extended_length)
    
    return noise

def add_noise_to_signal(signal, snr_db):
    # Расчет мощности сигнала
    signal_power = np.mean(signal ** 2)
    
    # Генерация шума
    noise = generate_awgn(snr_db, signal_power)
    
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

print("Генерация ПСП L1OF ГЛОНАСС")
print("=" * 60)

# Генерируем ПСП
l1of_code = generate_l1of()

# Проверяем начальную последовательность
first_9_chips = output_of_elements(l1of_code, 9)
print(f"Первые 9 чипов ПСП: {first_9_chips}")
print(f"Ожидается: 111111100")

# Увеличиваем время накопления
accumulation_factor = 5
print(f"\nВремя накопления увеличено в {accumulation_factor} раз")

# Создаем расширенный ПСП (повторяем несколько периодов)
l1of_code_extended = np.tile(l1of_code, accumulation_factor)
extended_length = len(l1of_code_extended)

print(f"Новая длина сигнала: {extended_length} отсчетов")
print(f"Время накопления: {extended_length/chip_rate*1000:.1f} мс")

random_delay = np.random.randint(0, code_length)  # 0-510 отсчетов (один период)
random_doppler = np.random.randint(-5000, 5000)   # -5кГц до +5кГц

print(f"\nСлучайные параметры:")
print(f"Задержка: {random_delay} отсчетов ({random_delay/chip_rate*1e6:.2f} мкс)")
print(f"Доплер: {random_doppler} Гц")

# Создаем сигнал с задержкой и доплером на расширенной длительности
t_extended = np.arange(extended_length) / chip_rate

phase_shift_extended = 2 * np.pi * random_doppler * t_extended

# Задержка по времени и частоте применяется ко всему расширенному сигналу
delayed_clean_signal_extended = np.roll(l1of_code_extended, random_delay)
delayed_clean_signal_extended = delayed_clean_signal_extended * np.exp(-1j * phase_shift_extended)

# Добавляем шум к расширенному сигналу
noisy_delayed_signal_extended, noise_final_extended = add_noise_to_signal(np.real(delayed_clean_signal_extended), SNR)

complex_received_signal = noisy_delayed_signal_extended.astype(complex)

# Визуализация для проверки
plt.figure(figsize=(12, 6))

# График 1: Сравнение опорного сигнала и сигнала с задержкой/доплером + шум
plt.subplot(2, 1, 1)
plt.plot(l1of_code_extended[:50], 'b-', linewidth=2, label='Опорный ПСП')
plt.plot(noisy_delayed_signal_extended[:50], 'r-', alpha=0.7, label='Сигнал с задержкой + доплер + шум')
plt.title('Сравнение опорного сигнала и искаженного сигнала')
plt.ylabel('Амплитуда')
plt.legend()
plt.grid(True)

# График 2: Шумовая компонента
plt.subplot(2, 1, 2)
# Вычисляем шум как разность между зашумленным и чистым искаженным сигналом
noise_component = noisy_delayed_signal_extended[:50] - np.real(delayed_clean_signal_extended[:50])
plt.plot(noise_component, 'g-', label='Шум')
plt.title('Шумовая компонента')
plt.xlabel('Отсчеты')
plt.ylabel('Амплитуда')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()





# Параметры поиска

# Параметры поиска для расширенного сигнала
chip_duration = 1 / chip_rate  # длительность одного чипа ~1.96 мкс

# Шаг по времени
delta_T_seconds = 2e-6  # шаг 2 мкс по времени

# Шаг по частоте 
delta_F = 50  # Шаг по частоте 50 Гц

f_range = 5000  # ±5 кГц диапазон доплера

# Диапазон по времени
t_range_seconds = T_prn  # 1 мс диапазон задержек (полный период ПСП)

# Расчет количества точек поиска
num_freq_points = int(2 * f_range / delta_F) + 1
num_time_points = int(t_range_seconds / delta_T_seconds) + 1  # от 0 до T_prn с шагом delta_T

print(f"\nПараметры поиска для расширенного сигнала:")
print(f"Длительность чипа: {chip_duration*1e6:.2f} мкс")
print(f"Длительность ПСП: {T_prn*1e3:.1f} мс")
print(f"Шаг по задержке: {delta_T_seconds*1e6:.2f} мкс")
print(f"Шаг по частоте: {delta_F} Гц")
print(f"Диапазон доплера: ±{f_range} Гц")
print(f"Диапазон задержек: 0...{t_range_seconds*1e3:.1f} мс")
print(f"Кол-во точек поиска: {num_freq_points} по частоте, {num_time_points} по времени")
print(f"Общее количество гипотез: {num_freq_points * num_time_points}")





# ПОСЛЕДОВАТЕЛЬНЫЙ ПОИСК

def sequential_search_optimized(received_signal, reference_code):
    # Преобразуем код в формат ±1 для корреляции
    prn_code = 2 * reference_code - 1
    
    # Массивы для хранения результатов
    correlation_grid = np.zeros((num_freq_points, num_time_points))
    freq_grid = np.zeros(num_freq_points)
    delay_time_grid = np.zeros(num_time_points)  
    
    max_correlation = 0
    best_doppler = 0
    best_delay_time = 0  
    
    print("\nНачало последовательного поиска")
    start_time = time.time()
    search_count = 0
    
    # Перебор по частоте Доплера
    for i in range(num_freq_points):
        current_freq = -f_range + i * delta_F
        freq_grid[i] = current_freq
        
        # Перебор по времени 
        for j in range(num_time_points):
            current_delay_time = j * delta_T_seconds 
            current_delay_samples = int(current_delay_time * chip_rate)  
            
            delay_time_grid[j] = current_delay_time
            
            # Компенсация задержки и доплера в принятом сигнале
            t = np.arange(len(received_signal)) / chip_rate
            
            # 1. Компенсируем доплер
            doppler_compensated = received_signal * np.exp(1j * 2 * np.pi * current_freq * t)
            
            # 2. Компенсируем задержку 
            fully_compensated = np.roll(doppler_compensated, -current_delay_samples)
            
            # Вычисляем корреляцию с опорным кодом
            correlation = np.abs(np.dot(fully_compensated, prn_code))
            correlation_grid[i, j] = correlation
            
            search_count += 1
            
            # Обновляем максимум
            if correlation > max_correlation:
                max_correlation = correlation
                best_doppler = current_freq
                best_delay_time = current_delay_time 
    
    search_time = time.time() - start_time
    
    # Пересчитываем найденную задержку в отсчеты для вывода
    best_delay_samples = int(best_delay_time * chip_rate)
    
    print(f"\nРезультаты последовательного поиска:")
    print(f"Всего проверено гипотез: {search_count}")
    print(f"Время поиска: {search_time:.3f} сек")
    print(f"Найденная задержка: {best_delay_time*1e6:.2f} мкс ({best_delay_samples} отсчетов)")
    print(f"Найденный доплер: {best_doppler} Гц")
    print(f"Максимальная корреляция: {max_correlation:.2f}")
    
    return best_delay_samples, best_doppler, max_correlation, freq_grid, delay_time_grid, correlation_grid

# Запускаем последовательный поиск
print("\n" + "="*60)
print("ПОСЛЕДОВАТЕЛЬНЫЙ ПОИСК")
print("="*60)

found_delay_samples, found_doppler_opt, max_corr_opt, freq_grid_opt, delay_time_grid_opt, corr_grid_opt = sequential_search_optimized(
    complex_received_signal, l1of_code_extended 
)



# Визуализация для последовательного поиска 

# Окно 1: 2D карта корреляции
plt.figure(figsize=(10, 6))
plt.contourf(delay_time_grid_opt * 1e6, freq_grid_opt / 1000, corr_grid_opt, levels=30, cmap='viridis')
plt.colorbar(label='Корреляция')
plt.xlabel('Задержка (мкс)')
plt.ylabel('Доплер (кГц)')
plt.title('2D карта корреляции')
plt.axvline(x=random_delay/chip_rate*1e6, color='red', linestyle='--', label='Истинная задержка')
plt.axhline(y=random_doppler/1000, color='red', linestyle='--', label='Истинный доплер')
plt.scatter(found_delay_samples/chip_rate*1e6, found_doppler_opt/1000, color='yellow', s=100, marker='*', label='Найденный максимум')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Окно 2: 3D поверхность корреляции
fig = plt.figure(figsize=(10, 7))
ax_3d = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(delay_time_grid_opt * 1e6, freq_grid_opt / 1000)
surf = ax_3d.plot_surface(X, Y, corr_grid_opt, cmap='viridis', alpha=0.8)
ax_3d.scatter(found_delay_samples/chip_rate*1e6, found_doppler_opt/1000, max_corr_opt, 
             color='red', s=50, marker='o', label='Вершина')
ax_3d.set_xlabel('Задержка (мкс)')
ax_3d.set_ylabel('Доплер (кГц)')
ax_3d.set_zlabel('Корреляция')
ax_3d.view_init(elev=20, azim=45)
fig.colorbar(surf, label='Корреляция')
plt.tight_layout()
plt.show()

# Проверка точности
delay_error_seconds = abs(found_delay_samples/chip_rate - random_delay/chip_rate)
doppler_error_hz = abs(found_doppler_opt - random_doppler)

delay_accuracy = "ВЫСОКАЯ точность" if delay_error_seconds <= delta_T_seconds else "СРЕДНЯЯ точность" if delay_error_seconds <= 2*delta_T_seconds else "НИЗКАЯ точность"
doppler_accuracy = "ВЫСОКАЯ точность" if doppler_error_hz <= delta_F else "СРЕДНЯЯ точность" if doppler_error_hz <= 2*delta_F else "НИЗКАЯ точность"

print(f"\nТочность последовательного поиска:")
print(f"Задержка: {delay_accuracy} (ошибка {delay_error_seconds*1e6:.2f} мкс при шаге {delta_T_seconds*1e6:.2f} мкс)")
print(f"Доплер: {doppler_accuracy} (ошибка {doppler_error_hz} Гц при шаге {delta_F} Гц)")
print(f"Истинная задержка: {random_delay/chip_rate*1e6:.2f} мкс, найдена: {found_delay_samples/chip_rate*1e6:.2f} мкс")
print(f"Истинный доплер: {random_doppler} Гц, найден: {found_doppler_opt} Гц")






# ПАРАЛЛЕЛЬНЫЙ ПОИСК ПО ЗАДЕРЖКЕ НА ОСНОВЕ БПФ

def fft_delay_search(signal, code):
    # Преобразуем код в ±1
    code = 2 * code - 1
    
    # Длина сигнала
    N = len(signal)
    
    # Вычисляем БПФ опорного кода (с дополнением нулями для линейной свертки)
    code_fft = np.fft.fft(code, n=N)             # Дополняем до длины сигнала
    code_fft_conj = np.conj(code_fft)            # Комплексное сопряжение
    
    max_corr = 0
    best_delay = 0
    best_freq = 0
    
    start_time = time.time()
    
    doppler_range = np.arange(-f_range, f_range + delta_F, delta_F)
    
    # Перебор по частотам Доплера
    for freq in doppler_range:
        
        # Компенсируем доплер
        t = np.arange(N) / chip_rate
        compensated = signal * np.exp(-1j * 2 * np.pi * freq * t)
        
        # БПФ корреляция (линейная свертка через частотную область)
        signal_fft = np.fft.fft(compensated)
        product = signal_fft * code_fft_conj
        correlation = np.abs(np.fft.ifft(product))
        
        # Ищем максимум в пределах одного периода ПСП
        max_search_range = min(N, code_length)
        correlation_limited = correlation[:max_search_range]
        
        delay = np.argmax(correlation_limited)
        corr_value = correlation_limited[delay]
        
        # Запоминаем лучший результат
        if corr_value > max_corr:
            max_corr = corr_value
            best_delay = delay
            best_freq = freq
    
    search_time = time.time() - start_time
    
    print(f"\nРезультаты БПФ-поиска по задержке:")
    print(f"Время поиска: {search_time:.3f} сек")
    print(f"Найденная задержка: {best_delay} отсчетов ({best_delay/chip_rate*1e6:.2f} мкс)")
    print(f"Найденный доплер: {best_freq} Гц")
    print(f"Максимальная корреляция: {max_corr:.2f}")
    
    return best_delay, best_freq, max_corr

# Запускаем БПФ-поиск по задержке
print("\n" + "="*60)
print("БПФ-ПОИСК ПО ЗАДЕРЖКЕ")
print("="*60)

found_delay_fft, found_doppler_fft, max_corr_fft = fft_delay_search(
    complex_received_signal, l1of_code_extended
)



# Визуализация для БПФ-поиска 

# Окно 1: 2D карта корреляции
plt.figure(figsize=(10, 6))
plt.contourf(delay_time_grid_opt * 1e6, freq_grid_opt / 1000, corr_grid_opt, levels=30, cmap='viridis')
plt.colorbar(label='Корреляция')
plt.xlabel('Задержка (мкс)')
plt.ylabel('Доплер (кГц)')
plt.title('2D карта корреляции (БПФ-поиск)')
plt.axvline(x=random_delay/chip_rate*1e6, color='red', linestyle='--', label='Истинная задержка')
plt.axhline(y=random_doppler/1000, color='red', linestyle='--', label='Истинный доплер')
plt.scatter(found_delay_fft/chip_rate*1e6, found_doppler_fft/1000, color='yellow', s=100, marker='*', label='Найденный максимум')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Окно 2: 3D поверхность корреляции
fig = plt.figure(figsize=(10, 7))
ax_3d = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(delay_time_grid_opt * 1e6, freq_grid_opt / 1000)
surf = ax_3d.plot_surface(X, Y, corr_grid_opt, cmap='viridis', alpha=0.8)
ax_3d.scatter(found_delay_fft/chip_rate*1e6, found_doppler_fft/1000, max_corr_fft, 
             color='red', s=50, marker='o', label='Вершина')
ax_3d.set_xlabel('Задержка (мкс)')
ax_3d.set_ylabel('Доплер (кГц)')
ax_3d.set_zlabel('Корреляция')
ax_3d.view_init(elev=20, azim=45)
fig.colorbar(surf, label='Корреляция')
plt.tight_layout()
plt.show()

# Проверка точности БПФ-поиска
delay_error_seconds_fft = abs(found_delay_fft/chip_rate - random_delay/chip_rate)
doppler_error_hz_fft = abs(found_doppler_fft - random_doppler)

delay_accuracy_fft = "ВЫСОКАЯ точность" if delay_error_seconds_fft <= delta_T_seconds else "СРЕДНЯЯ точность" if delay_error_seconds_fft <= 2*delta_T_seconds else "НИЗКАЯ точность"
doppler_accuracy_fft = "ВЫСОКАЯ точность" if doppler_error_hz_fft <= delta_F else "СРЕДНЯЯ точность" if doppler_error_hz_fft <= 2*delta_F else "НИЗКАЯ точность"

print(f"\nТочность БПФ-поиска:")
print(f"Задержка: {delay_accuracy_fft} (ошибка {delay_error_seconds_fft*1e6:.2f} мкс при шаге {delta_T_seconds*1e6:.2f} мкс)")
print(f"Доплер: {doppler_accuracy_fft} (ошибка {doppler_error_hz_fft} Гц при шаге {delta_F} Гц)")





# ПАРАЛЛЕЛЬНЫЙ ПОИСК ПО ЧАСТОТЕ НА ОСНОВЕ БПФ

def fft_frequency_search(signal, code):
    # Преобразуем код в ±1
    code = 2 * code - 1
    
    # Длина сигнала
    N = len(signal)
    
    max_corr = 0
    best_delay = 0
    best_freq = 0
    
    print("Начало БПФ-поиска по частоте...")
    start_time = time.time()
    
    delta_T_samples = int(delta_T_seconds * chip_rate)
    
    delay_range = np.arange(0, code_length, delta_T_samples) 
    
    # Перебор по задержкам (последовательный)
    for delay in delay_range:
        
        # Компенсируем задержку
        delayed_signal = np.roll(signal, -delay)
        
        # Умножаем на опорный код 
        despread_signal = delayed_signal * code
        
        # БПФ для поиска частоты
        spectrum = np.fft.fft(despread_signal)
        power_spectrum = np.abs(spectrum[:N//2])
        
        # Находим пик в спектре (доплеровскую частоту)
        max_freq_idx = np.argmax(power_spectrum)
        freq = max_freq_idx * (chip_rate / N)
        
        # Значение корреляции = амплитуда пика
        corr_value = power_spectrum[max_freq_idx]
        
        # Запоминаем лучший результат
        if corr_value > max_corr:
            max_corr = corr_value
            best_delay = delay
            best_freq = freq
    
    search_time = time.time() - start_time
    
    print(f"\nРезультаты БПФ-поиска по частоте:")
    print(f"Время поиска: {search_time:.3f} сек")
    print(f"Найденная задержка: {best_delay} отсчетов ({best_delay/chip_rate*1e6:.2f} мкс)")
    print(f"Найденный доплер: {best_freq:.1f} Гц")
    print(f"Максимальная корреляция: {max_corr:.2f}")
    
    return best_delay, best_freq, max_corr

# Запускаем БПФ-поиск по частоте
print("\n" + "="*60)
print("БПФ-ПОИСК ПО ЧАСТОТЕ")
print("="*60)

found_delay_freq, found_doppler_freq, max_corr_freq = fft_frequency_search(
    complex_received_signal, l1of_code_extended
)



# Визуализация для БПФ-поиска по частоте 

# Окно 1: 2D карта корреляции
plt.figure(figsize=(10, 6))
plt.contourf(delay_time_grid_opt * 1e6, freq_grid_opt / 1000, corr_grid_opt, levels=30, cmap='viridis')
plt.colorbar(label='Корреляция')
plt.xlabel('Задержка (мкс)')
plt.ylabel('Доплер (кГц)')
plt.title('2D карта корреляции (БПФ-поиск по частоте)')
plt.axvline(x=random_delay/chip_rate*1e6, color='red', linestyle='--', label='Истинная задержка')
plt.axhline(y=random_doppler/1000, color='red', linestyle='--', label='Истинный доплер')
plt.scatter(found_delay_freq/chip_rate*1e6, found_doppler_freq/1000, color='yellow', s=100, marker='*', label='Найденный максимум')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Окно 2: 3D поверхность корреляции
fig = plt.figure(figsize=(10, 7))
ax_3d = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(delay_time_grid_opt * 1e6, freq_grid_opt / 1000)
surf = ax_3d.plot_surface(X, Y, corr_grid_opt, cmap='viridis', alpha=0.8)
ax_3d.scatter(found_delay_freq/chip_rate*1e6, found_doppler_freq/1000, max_corr_freq, 
             color='red', s=50, marker='o', label='Вершина')
ax_3d.set_xlabel('Задержка (мкс)')
ax_3d.set_ylabel('Доплер (кГц)')
ax_3d.set_zlabel('Корреляция')
ax_3d.view_init(elev=20, azim=45)
fig.colorbar(surf, label='Корреляция')
plt.tight_layout()
plt.show()

# Проверка точности БПФ-поиска по частоте
delay_error_seconds_freq = abs(found_delay_freq/chip_rate - random_delay/chip_rate)
doppler_error_hz_freq = abs(found_doppler_freq - random_doppler)

delay_accuracy_freq = "ВЫСОКАЯ точность" if delay_error_seconds_freq <= delta_T_seconds else "СРЕДНЯЯ точность" if delay_error_seconds_freq <= 2*delta_T_seconds else "НИЗКАЯ точность"
doppler_accuracy_freq = "ВЫСОКАЯ точность" if doppler_error_hz_freq <= delta_F else "СРЕДНЯЯ точность" if doppler_error_hz_freq <= 2*delta_F else "НИЗКАЯ точность"

print(f"\nТочность БПФ-поиска по частоте:")
print(f"Задержка: {delay_accuracy_freq} (ошибка {delay_error_seconds_freq*1e6:.2f} мкс при шаге {delta_T_seconds*1e6:.2f} мкс)")
print(f"Доплер: {doppler_accuracy_freq} (ошибка {doppler_error_hz_freq:.1f} Гц при шаге {delta_F} Гц)")





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