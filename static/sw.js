// Service Worker for AgroPredict AI
// Cache-static assets for offline use and enable offline-first strategy for navigation.

const CACHE_NAME = 'agropredict-cache-v1';
const APP_SHELL = [
  '/',
  '/index.html',
  '/leafdoctor.html',
  '/about.html',
  '/static/styles.css',
  '/static/manifest.json',
  '/static/sw.js',
  'https://cdn.tailwindcss.com?plugins=typography',
  'https://cdn.jsdelivr.net/npm/axios/dist/axios.min.js',
  'https://cdn.jsdelivr.net/npm/marked/marked.min.js',
  'https://cdn.jsdelivr.net/npm/choices.js/public/assets/scripts/choices.min.js',
  'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css'
];

self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => cache.addAll(APP_SHELL))
  );
  self.skipWaiting();
});

self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(keys => Promise.all(
      keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k))
    ))
  );
  self.clients.claim();
});

self.addEventListener('fetch', event => {
  const { request } = event;
  if (request.method !== 'GET') return;

  // Network-first strategy for API calls to ensure fresh data
  if (request.url.includes('/predict') || request.url.includes('/crops') || request.url.includes('/translate')) {
    event.respondWith(
      fetch(request)
        .then(response => response)
        .catch(() => caches.match(request))
    );
    return;
  }

  // Cache-first strategy for all other requests
  event.respondWith(
    caches.match(request).then(cachedResponse => {
      if (cachedResponse) {
        return cachedResponse;
      }
      return fetch(request).then(networkResponse => {
        return caches.open(CACHE_NAME).then(cache => {
          cache.put(request, networkResponse.clone());
          return networkResponse;
        });
      });
    })
  );
});
