// controllers/userController.js
const db = require('../db');

exports.login = (req, res) => {
  const { username, password } = req.body;
  
  const sql = 'SELECT * FROM users WHERE username = ? AND password = ?';
  db.query(sql, [username, password], (err, results) => {
    if (err) return res.status(500).send({ error: 'Database error' });
    
    if (results.length > 0) {
      const user = results[0];
      res.send({ 
        message: 'Login successful', 
        user: {
          id: user.id,
          username: user.username,
          userType: user.role
        }
      });
    } else {
      res.status(401).send({ message: 'Invalid credentials' });
    }
  });
};



exports.register = (req, res) => {
  const { username, password, role } = req.body;
  
  const sql = 'INSERT INTO users (username, password, role) VALUES (?, ?, ?)';
  db.query(sql, [username, password, role], (err, result) => {
    if (err) {
      if (err.code === 'ER_DUP_ENTRY') {
        return res.status(400).send({ error: 'Username already exists' });
      }
      return res.status(500).send({ error: 'Registration failed' });
    }
    
    res.send({ 
      message: 'User registered successfully',
      user: {
        id: result.insertId,
        username,
        userType: role
      }
    });
  });
};
