const jwt = require('jsonwebtoken');
const { jwtSecret } = require('../config/config');

function verifyToken(req, res, next) {
  // Get token from the "Authorization" header
  const authHeader = req.headers['authorization'];
  if (!authHeader) {
    return res.status(403).json({ message: "No token provided." });
  }

  // Expecting the header to be in the format: "Bearer <token>"
  const parts = authHeader.split(" ");
  if (parts.length !== 2) {
    return res.status(403).json({ message: "Token format is invalid." });
  }
  const token = parts[1];

  // Verify token using jwtSecret
  jwt.verify(token, jwtSecret, (err, decoded) => {
    if (err) {
      return res.status(401).json({ message: "Failed to authenticate token." });
    }
    req.user = decoded;
    next();
  });
}

module.exports = { verifyToken };
