// db.js
const mysql = require('mysql2');

const connection = mysql.createConnection({
  host: 'localhost',
  user: 'demo',              // Update as needed
  password: 'allubb', // Update with your MySQL password
  database: 'ad4'
});

connection.connect((err) => {
  if (err) {
    console.error('Error connecting to MySQL database: ', err);
    process.exit(1);
  }
  console.log('Connected to MySQL database');
});

module.exports = connection;
