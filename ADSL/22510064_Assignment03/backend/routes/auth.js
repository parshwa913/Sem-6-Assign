const express = require('express');
const router = express.Router();
const db = require('../config/db');
const jwt = require('jsonwebtoken');
const { jwtSecret } = require('../config/config');

router.post('/login', async (req, res) => {
  const { username, password, role } = req.body;
  console.log('Login attempt:', req.body);
  
  try {
    let query = '';
    let params = [];
    const roleLower = role.toLowerCase();

    if (roleLower === 'admin') {
      query = 'SELECT * FROM admin_users WHERE username = ?';
      params = [username];
    } 
    else if (roleLower === 'instructor') {
      query = 'SELECT * FROM instructor_users WHERE username = ?';
      params = [username];
    } 
    else if (roleLower === 'student') {
      query = 'SELECT * FROM student_users WHERE username = ?';
      params = [username];
    } 
    else {
      return res.status(400).json({ message: 'Invalid role specified.' });
    }

    console.log(`Executing query for ${roleLower}:`, query, 'with params:', params);
    const [rows] = await db.execute(query, params);
    console.log('Query result:', rows);

    if (rows.length === 0) {
      return res.status(404).json({ message: 'User not found.' });
    }

    const user = rows[0];

    if (user.password !== password) {
      return res.status(401).json({ message: 'Invalid password.' });
    }

    let payload = {};
    if (roleLower === 'admin') {
      payload = { id: user.admin_id, role: 'admin' };
    } else if (roleLower === 'instructor') {
      payload = { id: user.instructor_id, role: 'instructor' };
    } else if (roleLower === 'student') {
      payload = { id: user.student_id, role: 'student' };
    }

    const token = jwt.sign(payload, jwtSecret, { expiresIn: '8h' });
    return res.json({ token, role: roleLower, user });
  } catch (err) {
    console.error('Login error:', err);
    res.status(500).json({ message: 'Server error during login.' });
  }
});

module.exports = router;
