tuples_of_diam_the_real = [(266, 247.9), (248, 229.62), (254, 235.46)]

# Calculate average error
percent_errors = [
    abs((real - theoretical) / theoretical) * 100
    for real, theoretical in tuples_of_diam_the_real
]
avg_error_percent = sum(percent_errors) / len(percent_errors)
avg_error_decimal = avg_error_percent / 100

# Ask for a real value
real_input = float(input("Enter your desired real value: "))

# Estimate theoretical value
estimated_theoretical = real_input / (1 + avg_error_decimal)

print(f"Average error: {avg_error_percent:.2f}%")
print(f"Estimated theoretical value: {estimated_theoretical:.2f}")