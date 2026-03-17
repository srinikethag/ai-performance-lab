import http from 'k6/http';
import { sleep } from 'k6';

export const options = {
  stages: [
    { duration: '30s', target: 1 },   // baseline
    { duration: '30s', target: 5 },   // light load
    { duration: '30s', target: 10 },  // moderate load
    { duration: '30s', target: 20 },  // heavy load
    { duration: '30s', target: 0 },   // cool down
  ],
};

const url = 'http://localhost:8000/generate';

export default function () {

  const payload = JSON.stringify({
    prompt: "Explain transformers in simple terms",
    model: "llama3.2:latest"
  });

  const params = {
    headers: {
      'Content-Type': 'application/json',
    },
  };

  const res = http.post(url, payload, {
    headers: { 'Content-Type': 'application/json' },
    timeout: '120s'
    });

  if (res.status !== 200) {
    console.error(`Request failed: ${res.status}`);
  }

  sleep(1);
}