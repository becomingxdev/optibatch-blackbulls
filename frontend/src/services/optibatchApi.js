import apiClient from './apiClient';

export const predictBatch = async (batchParameters) => {
  try {
    const response = await apiClient.post('/predict', batchParameters);
    return response.data;
  } catch (error) {
    console.error('Failed to predict batch:', error);
    throw error;
  }
};

export const optimizeBatch = async (batchParameters, predictedMetrics) => {
  try {
    const response = await apiClient.post('/optimize', {
      batch_parameters: batchParameters,
      predicted_metrics: predictedMetrics || {},
    });
    return response.data;
  } catch (error) {
    console.error('Failed to optimize batch:', error);
    throw error;
  }
};

export const monitorBatch = async (batchParameters) => {
  try {
    const response = await apiClient.post('/monitor', { batch_parameters: batchParameters });
    return response.data;
  } catch (error) {
    console.error('Failed to monitor batch:', error);
    throw error;
  }
};

export const simulateBatch = async (batchParameters) => {
  try {
    const response = await apiClient.post('/simulate', batchParameters);
    return response.data;
  } catch (error) {
    console.error('Failed to simulate batch:', error);
    throw error;
  }
};

export const runParameterSweep = async (parameterRanges, numSimulations) => {
  try {
    const response = await apiClient.post('/sweep', { parameter_ranges: parameterRanges, num_simulations: numSimulations });
    return response.data;
  } catch (error) {
    console.error('Failed to run parameter sweep:', error);
    throw error;
  }
};

export const runSweep = async (payload) => {
  try {
    const response = await apiClient.post('/sweep', {
      parameter_ranges: {
        temperature: payload.temperature_range,
        pressure: payload.pressure_range,
      },
      num_simulations: payload.steps
    });
    return response.data;
  } catch (error) {
    console.error('Failed to run sweep:', error);
    throw error;
  }
};

export const fetchSummary = async (strategy) => {
  try {
    const response = await apiClient.get(`/api/data/summary?strategy=${strategy}`);
    return response.data;
  } catch (error) {
    console.error('Failed to fetch summary:', error);
    throw error;
  }
};

export const chatWithAi = async (message) => {
  try {
    const response = await apiClient.post('/api/chat', { message });
    return response.data;
  } catch (error) {
    console.error('Failed to chat:', error);
    throw error;
  }
};
