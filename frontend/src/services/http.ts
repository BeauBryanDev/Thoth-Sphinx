// ============================================================
// Axios HTTP Client — single configured instance
// ============================================================

import axios, {
  type AxiosInstance,
  type AxiosRequestConfig,
  type AxiosResponse,
  type InternalAxiosRequestConfig,
} from "axios";
import type { ApiResponse } from "@/types";

// The base URL is read from an env var at build time (Vite convention).
// Falls back to a placeholder so the skeleton runs without a backend.
const BASE_URL =
  (import.meta.env.VITE_API_BASE_URL as string | undefined) ??
  "https://api.sphinxeyes.example.com";

const DEFAULT_TIMEOUT = 15_000;

// ---------- instance ----------
export const http: AxiosInstance = axios.create({
  baseURL: BASE_URL,
  timeout: DEFAULT_TIMEOUT,
  headers: {
    "Content-Type": "application/json",
    Accept: "application/json",
  },
});

// ---------- request interceptor (auth) ----------
http.interceptors.request.use(
  (config: InternalAxiosRequestConfig) => {
    // Attach a bearer token if one exists in storage.
    const token = typeof window !== "undefined"
      ? window.localStorage.getItem("sphinx_token")
      : null;
    if (token) {
      config.headers = config.headers ?? {};
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error),
);

// ---------- response interceptor (error normalisation) ----------
http.interceptors.response.use(
  (response: AxiosResponse) => response,
  (error) => {
    // Show the server's own explanation (guardrails: daily quota, rate window, budget, 502 error_id)
    // instead of axios's "Request failed with status code 429". 422 details are lists; keep those as is.
    const detail = error?.response?.data?.detail;
    if (typeof detail === "string" && detail) {
      error.message = detail;
    }
    // eslint-disable-next-line no-console
    console.error("[SphinxEyes API error]", error?.message ?? error);
    return Promise.reject(error);
  },
);

// ---------- typed helper ----------
/** Perform a request and unwrap the standard ApiResponse<T> envelope. */
export const request = async <T>(
  config: AxiosRequestConfig,
): Promise<T> => {
  const { data } = await http.request<ApiResponse<T>>(config);
  return data.data;
};

export default http;
