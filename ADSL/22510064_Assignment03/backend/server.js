const express = require('express');
const bodyParser = require('body-parser');
const cors = require('cors');
const { verifyToken } = require('./middleware/authMiddleware'); 

const app = express();

app.use(cors());
app.use(bodyParser.json());

const authRoutes = require('./routes/auth');
const instructorRoutes = require('./routes/instructors');
const studentRoutes = require('./routes/students');
const coursesRoutes = require('./routes/courses');
const reportRoutes = require('./routes/report');
app.use('/api/report', verifyToken, reportRoutes);

app.use('/api/auth', authRoutes);
app.use('/api/instructors', verifyToken, instructorRoutes);
app.use('/api/students', verifyToken, studentRoutes);
app.use('/api/courses', verifyToken, coursesRoutes);

// Root route
app.get('/', (req, res) => {
  res.send('Welcome to the MIS Portal API');
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
