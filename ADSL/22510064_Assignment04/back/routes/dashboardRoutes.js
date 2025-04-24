// routes/dashboardRoutes.js
const express = require('express');
const router = express.Router();
const dashboardController = require('../controllers/dashboardController');

router.get('/dashboard/teacher/:teacherId', dashboardController.getTeacherDashboard);

router.get('/student', dashboardController.getStudentDashboard);
router.get('/exams/recent', dashboardController.getRecentExams);
router.get('/dashboard/performance', dashboardController.getStudentPerformance);
router.get('/notifications/teacher', dashboardController.getTeacherNotifications);
router.get('/exams/upcoming', dashboardController.getUpcomingExams);
router.get('/dashboard/student-performance', dashboardController.getStudentPerformanceStats);
router.get('/notifications/student', dashboardController.getStudentNotifications);

module.exports = router;
