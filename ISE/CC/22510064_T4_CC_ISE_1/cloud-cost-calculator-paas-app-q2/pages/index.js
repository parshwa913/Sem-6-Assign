import Head from 'next/head';
import { useState } from 'react';

export default function Home() {
  const [storage, setStorage] = useState('');
  const [cpu, setCpu] = useState('');
  const [bandwidth, setBandwidth] = useState('');
  const [cost, setCost] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    // Call the API endpoint with the query parameters
    const res = await fetch(
      `/api/cost?storage=${storage}&cpu=${cpu}&bandwidth=${bandwidth}`
    );
    const data = await res.json();
    setCost(data.cost);
  };

  // Overall page style
  const pageStyle = {
    margin: 0,
    padding: 0,
    fontFamily: '"Roboto", sans-serif',
    backgroundColor: '#f8f9fa',
    color: '#333',
  };

  // Header style (dark blue)
  const headerStyle = {
    backgroundColor: '#232f3e',
    padding: '1rem 2rem',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    color: '#fff',
  };

  const logoStyle = {
    fontSize: '1.5rem',
    fontWeight: 'bold',
  };

  const navLinkStyle = {
    color: '#fff',
    textDecoration: 'none',
    marginLeft: '1.5rem',
  };

  // Hero section style
  const heroStyle = {
    backgroundImage: 'linear-gradient(to right, #131921, #232f3e)',
    color: '#fff',
    padding: '4rem 2rem',
    textAlign: 'center',
  };

  const heroHeadingStyle = {
    fontSize: '2.8rem',
    margin: '0 0 1rem',
  };

  const heroSubheadingStyle = {
    fontSize: '1.3rem',
    margin: '0 0 2rem',
  };

  // Form container style (card-like)
  const formContainerStyle = {
    backgroundColor: '#fff',
    margin: '2rem auto',
    padding: '2rem',
    borderRadius: '8px',
    boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
    maxWidth: '500px',
  };

  // Form group styles
  const formGroupStyle = {
    marginBottom: '1.5rem',
  };

  const labelStyle = {
    display: 'block',
    marginBottom: '0.5rem',
    fontWeight: '500',
    fontSize: '1rem',
  };

  const inputStyle = {
    width: '100%',
    padding: '0.75rem',
    fontSize: '1rem',
    borderRadius: '4px',
    border: '1px solid #ccc',
  };

  // Button style (using a vibrant orange)
  const buttonStyle = {
    backgroundColor: '#ff9900',
    color: '#fff',
    border: 'none',
    padding: '1rem',
    width: '100%',
    fontSize: '1.1rem',
    borderRadius: '4px',
    cursor: 'pointer',
    marginTop: '1rem',
  };

  const resultStyle = {
    marginTop: '2rem',
    textAlign: 'center',
    fontSize: '1.5rem',
    fontWeight: '600',
    color: '#232f3e',
  };

  // Footer style
  const footerStyle = {
    backgroundColor: '#232f3e',
    color: '#fff',
    padding: '1rem 2rem',
    textAlign: 'center',
    fontSize: '0.875rem',
  };

  return (
    <>
      <Head>
        <title>Parshwa's Cloud Calculator</title>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="true" />
        <link
          href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap"
          rel="stylesheet"
        />
      </Head>
      <div style={pageStyle}>
        {/* Header */}
        <header style={headerStyle}>
          <div style={logoStyle}>Parshwa's Cloud Calculator</div>
          <nav>
            <a href="#" style={navLinkStyle}>Home</a>
            <a href="#" style={navLinkStyle}>Pricing</a>
            <a href="#" style={navLinkStyle}>Docs</a>
          </nav>
        </header>

        {/* Hero Section */}
        <section style={heroStyle}>
          <h1 style={heroHeadingStyle}>Optimize Your Cloud Spend</h1>
          <p style={heroSubheadingStyle}>
            Estimate your cloud costs with precision and efficiency.
          </p>
        </section>

        {/* Form Section */}
        <div style={formContainerStyle}>
          <h2 style={{ textAlign: 'center', marginBottom: '1.5rem' }}>
            Cloud Cost Calculator
          </h2>
          <form onSubmit={handleSubmit}>
            <div style={formGroupStyle}>
              <label style={labelStyle}>Storage (GB):</label>
              <input
                type="number"
                value={storage}
                onChange={(e) => setStorage(e.target.value)}
                required
                style={inputStyle}
              />
            </div>
            <div style={formGroupStyle}>
              <label style={labelStyle}>CPU Cores:</label>
              <input
                type="number"
                value={cpu}
                onChange={(e) => setCpu(e.target.value)}
                required
                style={inputStyle}
              />
            </div>
            <div style={formGroupStyle}>
              <label style={labelStyle}>Bandwidth (TB):</label>
              <input
                type="number"
                value={bandwidth}
                onChange={(e) => setBandwidth(e.target.value)}
                required
                style={inputStyle}
              />
            </div>
            <button type="submit" style={buttonStyle}>
              Calculate Cost
            </button>
          </form>
          {cost !== null && (
            <div style={resultStyle}>
              Estimated Cloud Cost: ${cost}
            </div>
          )}
        </div>

        {/* Footer */}
        <footer style={footerStyle}>
          &copy; {new Date().getFullYear()} Parshwa's Cloud Calculator. All rights reserved.
        </footer>
      </div>
    </>
  );
}
