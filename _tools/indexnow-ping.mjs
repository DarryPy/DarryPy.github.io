#!/usr/bin/env node
// IndexNow ping — submit one or more URLs of darrypy.github.io to Bing/Yandex/Seznam/Naver
// in one call. Usage:
//   node _tools/indexnow-ping.mjs https://darrypy.github.io/posts/xxx.html
//   node _tools/indexnow-ping.mjs <url1> <url2> <url3> ...

const KEY = '8f432c1772450c962a07acf05dd4d8c9';
const HOST = 'darrypy.github.io';
const ENDPOINT = 'https://api.indexnow.org/IndexNow';

const urls = process.argv.slice(2).filter(Boolean);
if (urls.length === 0) {
  console.error('usage: node indexnow-ping.mjs <url> [<url2> ...]');
  process.exit(1);
}

const payload = {
  host: HOST,
  key: KEY,
  keyLocation: `https://${HOST}/${KEY}.txt`,
  urlList: urls,
};

const res = await fetch(ENDPOINT, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json; charset=utf-8' },
  body: JSON.stringify(payload),
});

const body = await res.text().catch(() => '');
console.log(`IndexNow ${res.status} ${res.statusText} | submitted ${urls.length} url(s)${body ? ' | body: ' + body.slice(0, 200) : ''}`);

// 200 OK / 202 Accepted = success
if (![200, 202].includes(res.status)) process.exit(1);
