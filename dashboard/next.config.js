/** @type {import('next').NextConfig} */
const nextConfig = {
  // Proxy API requests to the Python backend
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: "http://localhost:8780/api/:path*",
      },
    ];
  },
};

module.exports = nextConfig;
