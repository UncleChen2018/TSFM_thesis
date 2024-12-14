import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from collections import deque
import transformers
from chronos import ChronosPipeline


import gzip
import tarfile
import os
import re
import json


from collections import deque
import numpy as np
import torch
from tqdm import tqdm
import transformers

from collections import deque
import numpy as np
import torch
from tqdm.notebook import tqdm  # Import tqdm for notebook
import transformers


import tempfile
import shutil

import subprocess

import zstandard as zstd


default_config = {
    'EB': 0.001,    # Example default error bound
    'PDT': 1,       # Default prediction distance/time
    'model': 'base',# Default model type
    'seed': 42      # Default random seed
}




def rolling_forecast_and_reconstruct(
    initial_context, PDT, command, extra_data, EB=0.001,  model=0, seed=42
):
    # change all types to float32 (since the model's tensor is float32)
    initial_context = initial_context.astype(np.float32)
    extra_data = extra_data.astype(np.float32)
 
    if command not in ["compress", "decompress"]:
        raise ValueError("Invalid command. Choose either 'compress' or 'decompress'.")
    transformers.set_seed(seed)
    
    MODEL_sizes = ["tiny", "mini", "small", "base", "large"]
    model_to_use = MODEL_sizes[model]
    
    dynamic_context = deque(initial_context, maxlen=len(initial_context))
    pipeline = ChronosPipeline.from_pretrained(
        f"amazon/chronos-t5-{model_to_use}",
        device_map="cuda",
        torch_dtype=torch.float32,
    )

    # Calculate the size of arrays based on extra_data and PDT
    total_steps = len(extra_data)
    forecasts = np.zeros(total_steps)
    errors = np.zeros(total_steps) if command == "compress" else extra_data.copy()

    # Convert the initial dynamic context to a tensor once and update as tensor
    context_tensor = torch.tensor(list(dynamic_context), dtype=torch.float32)

    for i in tqdm(range(0, total_steps, PDT), desc=f"{command} processing", unit="step"):
        forecast_length = min(PDT, total_steps - i)
        forecast = pipeline.predict(
            context=context_tensor, prediction_length=forecast_length, num_samples=20
        )

        median_forecast = np.median(forecast[0].numpy(), axis=0)


        batch_slice = slice(i, i + forecast_length)
        actual_values = extra_data[batch_slice]

        if command == "compress":
            error = (actual_values - median_forecast).astype('float32')
            # rounded_errors = recover_error_approximation(get_error_quantiles(error, EB),EB) if EB is not None else error
            #print(error)
            #print(LinearQuantizer.quantize(error, EB))

            
            
            rounded_errors = LinearQuantizer.peek(error, EB) if EB is not None else error
            if not np.allclose(error, rounded_errors,atol = EB):
                print(i, error, rounded_errors, 'not work')
        else:
            rounded_errors = errors[batch_slice]

        forecasts[batch_slice] = median_forecast
        errors[batch_slice] = rounded_errors

        # Update context tensor for next batch
        adjusted_values = median_forecast + rounded_errors
        #print(adjusted_values.dtype)
        #print(i,median_forecast,rounded_errors,adjusted_values)
        dynamic_context.extend(
            adjusted_values[-len(dynamic_context) :]
        )  # Ensure context stays within the max length
        context_tensor = torch.tensor(list(dynamic_context), dtype=torch.float32)

    return forecasts.astype('float32'), errors.astype('float32')




def compute_error_metrics(errors, actual_values):
    """
    Compute various statistical metrics for the given errors, including relative MSE.

    Args:
    - errors (numpy.array): The array of forecast errors.
    - actual_values (numpy.array): The actual values from which the errors were calculated.

    Returns:
    - dict: A dictionary containing MSE, MAE, Min, Max, Range of errors, and Relative MSE.
    """
    mse = np.mean(np.square(errors))
    mae = np.mean(np.abs(errors))
    min_error = np.min(errors)
    max_error = np.max(errors)
    error_range = max_error - min_error
    
    # Compute the variance of the actual values
    variance = np.var(actual_values)
    
    # Compute Relative MSE
    relative_mse = mse / variance if variance != 0 else float('inf')  # Avoid division by zero

    return {
        "Mean Squared Error (MSE)": mse,
        "Mean Absolute Error (MAE)": mae,
        "Minimum Error": min_error,
        "Maximum Error": max_error,
        "Range of Errors": error_range,
        "Relative MSE": relative_mse,
    }

# Example usage:
# errors = np.array([...])  # Forecast errors
# actual_values = np.array([...])  # Actual values
# error_metrics = compute_error_metrics(errors, actual_values)
# print(error_metrics)



def plot_error_distribution(errors):
    """
    Generate plots to visualize the distribution of forecast errors.

    Args:
    - errors (numpy.array): The array of forecast errors.
    """
    plt.figure(figsize=(12, 6))

    # Histogram
    plt.subplot(1, 2, 1)
    plt.hist(errors, bins=50, color="blue", alpha=0.7)
    plt.title("Histogram of Errors")
    plt.xlabel("Error")
    plt.ylabel("Frequency")

    # Boxplot
    plt.subplot(1, 2, 2)
    plt.boxplot(errors, vert=False)
    plt.title("Boxplot of Errors")
    plt.xlabel("Error")

    plt.tight_layout()
    plt.show()




def plot_time_series_with_forecast(actual_array, forecast_array, ctx):
    """
    Plots the actual time series data along with the forecasted data on the same graph.
    Assumes forecast_array starts from the 'ctx' index of the actual_array.

    Args:
    actual_array (numpy.ndarray or pandas.Series): The actual time series data.
    forecast_array (numpy.ndarray): The forecasted data, starting from the 'ctx' index of the actual array.
    ctx (int): The index from which the forecast starts.
    """
    plt.figure(figsize=(10, 6))
    
    # Convert actual_array to numpy array if it's a pandas Series
    if isinstance(actual_array, pd.Series):
        actual_array = actual_array.values

    # Create index arrays for plotting
    actual_index = np.arange(len(actual_array))
    forecast_index = np.arange(ctx, ctx + len(forecast_array))

    # Plot the actual data
    plt.plot(actual_index, actual_array, label='Actual Data', linestyle='-', marker='', color='blue')
    
    # Plot the forecast data
    plt.plot(forecast_index, forecast_array, label='Forecast', linestyle='-', marker='', color='tomato')
    
    # Adding titles and labels
    plt.title('Time Series Forecast')
    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage:
# Assume `actual_data` and `forecast_data` are numpy arrays
# If you're starting with a pandas Series for actual data:
# actual_data = pd.Series(...).values or just pd.Series(...)
# forecast_data = np.array(...)
# ctx = index where forecast starts
# plot_time_series_with_forecast(actual_data, forecast_data, ctx)




class LinearQuantizer:
    @staticmethod
    def quantize(data, error_bound):
        
        quant_indices = np.zeros(
            data.shape, dtype=np.int16
        )  # Quantized indices as int16
        unpred = []  # List to hold values that cannot be quantized within int16 bounds

        for i, value in enumerate(data):
            # Calculate half index from the value and error bound, considering the sign
            half_index = (np.abs(value) / error_bound // 2 + 1) * (
                1 if value >= 0 else -1
            )
            #print(value, error_bound, np.abs(value) / error_bound)
            #print(np.abs(value) / error_bound)
            half_index = half_index.astype("int64")
            #print(half_index)

            # Check if half_index fits within int16 limits
            if -32768 <= half_index <= 32767:
                quant_indices[i] = np.int16(half_index)
            else:
                # Check if value fits within int32 before appending
                if np.iinfo(np.int32).min <= half_index <= np.iinfo(np.int32).max:
                    unpred.append(np.int32(half_index))  # Store directly as int32
                else:
                    raise ValueError(
                        f"Error: Value {value} out of int32 quantization range."
                    )
        # Convert unpred list to a np.int32 array before returning
        unpred = np.array(unpred, dtype=np.int32)
        return quant_indices, unpred

    @staticmethod
    def recover(quant_indices, unpred, error_bound):
        
        recovered_data = np.zeros(quant_indices.shape, dtype=np.float32)
        unpred_index = 0

        for i, index in enumerate(quant_indices):
            if index == 0:
                index = unpred[unpred_index]
                unpred_index += 1
            else:
                index = index.astype(np.int32)

            sign = 1 if index > 0 else -1
            recovered_data[i] = (np.abs(index) * 2 - 1) * error_bound * sign

        return recovered_data.astype(np.float32)

    @staticmethod
    def peek(data, error_bound):
        # Perform quantization and immediate recovery to preview the effect
        quantized_indices, unpred = LinearQuantizer.quantize(data, error_bound)
        recovered_data = LinearQuantizer.recover(quantized_indices, unpred, error_bound)
        return recovered_data



import numpy as np
import zstandard as zstd

def compress_data(model, EB, PDT, context, quant_indices, unpred):
    # Initialize the compressor
    compressor = zstd.ZstdCompressor(level=22)

    # Check and convert the model to uint8
    if not isinstance(model, int):
        raise ValueError("Model should be uint8")
    model = np.array([model], dtype=np.uint8)

    # Ensure EB is a float and convert to float64
    if not isinstance(EB, float):
        raise ValueError("EB should be float64")
    EB = np.array([EB], dtype=np.float64)

    # Check and convert PDT to uint8
    if not isinstance(PDT, int):
        raise ValueError("PDT should be uint8")
    PDT = np.array([PDT], dtype=np.uint8)

    # Validate and handle data types for context, quant_indices, and unpred
    if context.dtype != np.float32:
        raise ValueError("Context should be float32")
    if quant_indices.dtype != np.int16:
        raise ValueError("Quant indices should be int16")
    if unpred.dtype != np.int32:
        raise ValueError("Unpred should be int32")

    # Compress the individual components
    compressed_context = compressor.compress(context.tobytes())
    compressed_quant_indices = compressor.compress(quant_indices.tobytes())
    compressed_unpred = compressor.compress(unpred.tobytes())

    # Combine all components into a single byte stream for final compression
    data_to_compress = (
        model.tobytes()
        + EB.tobytes()
        + PDT.tobytes()
        + np.array(
            [len(compressed_context), len(compressed_quant_indices)], dtype=np.int64
        ).tobytes()
        + compressed_context
        + compressed_quant_indices
        + compressed_unpred
    )

    # Compress the combined data
    final_compressed_data = compressor.compress(data_to_compress)
    return final_compressed_data




# def compress_data(model, EB,context, PDT, quant_indices, unpred):
#     compressor = zstd.ZstdCompressor(level=22)
#     model = np.array([model], dtype=np.uint8)
    

#     # EB must be 64bits to prevent loss of precison 
#     EB = np.array([EB], dtype=np.float64)

#     PDT = np.array([PDT], dtype=np.uint8)

#     if 

#     if context.dtype != np.float32:
#         raise ValueError("Context should be float32")
#     if quant_indices.dtype != np.int16:
#         raise ValueError("Quant indices should be int16")
#     if unpred.dtype != np.int32:
#         raise ValueError("Unpred should be int32")
    

#     compressed_context = compressor.compress(context.tobytes())
#     compressed_quant_indices = compressor.compress(quant_indices.tobytes())
#     compressed_unpred = compressor.compress(unpred.tobytes())

#     data_to_compress = (
#         model.tobytes()
#         + EB.tobytes()
#         + np.array(
#             [len(compressed_context), len(compressed_quant_indices)], dtype=np.int64
#         ).tobytes()
#         + compressed_context
#         + compressed_quant_indices
#         + compressed_unpred
#     )

#     final_compressed_data = compressor.compress(data_to_compress)
#     return final_compressed_data

def decompress_data(compressed_data):
    decompressor = zstd.ZstdDecompressor()

    # Decompress the entire byte stream first
    data = decompressor.decompress(compressed_data)
    idx = 0

    # Read model (uint8)
    model = np.frombuffer(data[idx:idx + 1], dtype=np.uint8)[0]
    idx += 1

    # Read double precision EB (float64)
    EB = np.frombuffer(data[idx:idx + 8], dtype=np.float64)[0]
    idx += 8

    # Read Prediction Time Delta (PDT) (uint8)
    PDT = np.frombuffer(data[idx:idx + 1], dtype=np.uint8)[0]
    idx += 1

    PDT = int(PDT)  # change to int, other wise Chronos thorws error

    # Read the lengths of compressed context and quant_indices (int64)
    lengths = np.frombuffer(data[idx:idx + 2 * 8], dtype=np.int64)
    idx += 2 * 8

    # Decompress and read the context array (float32)
    context = np.frombuffer(
        decompressor.decompress(data[idx:idx + lengths[0]]), dtype=np.float32
    )
    idx += lengths[0]

    # Decompress and read the quant_indices array (int16)
    quant_indices = np.frombuffer(
        decompressor.decompress(data[idx:idx + lengths[1]]), dtype=np.int16
    )
    idx += lengths[1]

    # Decompress and read the unpred array (int32)
    unpred = np.frombuffer(decompressor.decompress(data[idx:]), dtype=np.int32)

    return model, EB, PDT, context, quant_indices, unpred


# def decompress_data(compressed_data):
#     decompressor = zstd.ZstdDecompressor()

#     data = decompressor.decompress(compressed_data)
#     idx = 0

#     # read model
#     model = np.frombuffer(data[idx : idx + 1], dtype=np.uint8)[0]
#     idx += 1
#     # read double presicion EB 
#     EB = np.frombuffer(data[idx : idx + 8], dtype=np.float64)[0]
#     idx += 8
#     lengths = np.frombuffer(data[idx : idx + 2 * 8], dtype=np.int64)
#     idx += 2 * 8

#     context = np.frombuffer(
#         decompressor.decompress(data[idx : idx + lengths[0]]), dtype=np.float32
#     )
#     idx += lengths[0]
#     quant_indices = np.frombuffer(
#         decompressor.decompress(data[idx : idx + lengths[1]]), dtype=np.int16
#     )
#     idx += lengths[1]
#     unpred = np.frombuffer(decompressor.decompress(data[idx:]), dtype=np.int32)

#     return model, EB, context, quant_indices, unpred



# 
def compress_to_file(data_array, file_path, CTX = 512, PDT=1,  EB=0.001, model=0):
    
    
    
    inital_context = (data_array[:CTX]).astype(np.float32)
    extra_data = data_array[CTX:].astype(np.float32)
    forecasts, errors = rolling_forecast_and_reconstruct(
    inital_context, PDT, "compress", extra_data, EB=EB, model=model)
    
    
    quant_indices, unpred = LinearQuantizer.quantize(errors, EB)
    compressed_data = compress_data(model, EB, PDT, inital_context, quant_indices, unpred)
    
    
    cr = len(data_array) * 4 / len(compressed_data)
    
    print(f"Compression ratio: {cr:.2f}")
    with open(file_path, "wb") as f:
        f.write(compressed_data)
        
    return compressed_data


def compress_to_file_debug(data_array, file_path, CTX = 512, PDT=1,  EB=0.001, model=0):
    
    
    
    inital_context = (data_array[:CTX]).astype(np.float32)
    extra_data = data_array[CTX:].astype(np.float32)
    forecasts, errors = rolling_forecast_and_reconstruct(
    inital_context, PDT, "compress", extra_data, EB=EB, model=model)
    
    
    quant_indices, unpred = LinearQuantizer.quantize(errors, EB)
    compressed_data = compress_data(model, EB, PDT, inital_context, quant_indices, unpred)
    
    
    cr = len(data_array) * 4 / len(compressed_data)
    
    print(f"Compression ratio: {cr:.2f}")
    with open(file_path, "wb") as f:
        f.write(compressed_data)
        
    return forecasts, errors



def decompress_to_csv(compressed_file_path, csv_file_path):
    with open(compressed_file_path, "rb") as f:
        compressed_data = f.read()
        
    model, EB, PDT, context, quant_indices, unpred = decompress_data(compressed_data)
    recovered_errors = LinearQuantizer.recover(quant_indices, unpred, EB)
    recovered_prediction, _ = rolling_forecast_and_reconstruct(
        context, PDT=PDT, command="decompress", extra_data=recovered_errors, EB=EB, model=model
    )
    np_full = np.hstack((context, recovered_prediction + recovered_errors))
    pd.DataFrame(np_full).to_csv(csv_file_path, index=False, header=False)
    return np_full


def decompress_to_mem(compressed_file_path):
    with open(compressed_file_path, "rb") as f:
        compressed_data = f.read()
   
    model, EB, PDT, context, quant_indices, unpred = decompress_data(compressed_data)
 
    recovered_errors = LinearQuantizer.recover(quant_indices, unpred, EB)
    recovered_prediction, _ = rolling_forecast_and_reconstruct(
        context, PDT=PDT, command="decompress", extra_data=recovered_errors, EB=EB, model=model
    )
    np_full = np.hstack((context, recovered_prediction + recovered_errors))
    return np_full, recovered_prediction, context,  recovered_errors

    

# # this part is for checking the validity of the reconstructed data and calculate CR

def check_data_valid(origainal_data, reconstructed_data,EB):
    return np.allclose(origainal_data, reconstructed_data, atol=EB)

def check_file_valid(original_file_path, reconstructed_file_path,EB):
    original_data = np.loadtxt(original_file_path, delimiter=",")
    reconstructed_data = np.loadtxt(reconstructed_file_path, delimiter=",")
    return check_data_valid(original_data, reconstructed_data,EB)