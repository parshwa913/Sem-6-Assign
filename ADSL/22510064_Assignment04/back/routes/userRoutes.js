// routes/userRoutes.js
const express = require('express');
const router = express.Router();
const userController = require('../controllers/userController');

router.post('/login', userController.login);
// Optional: router.post('/register', userController.register);

module.exports = router;
