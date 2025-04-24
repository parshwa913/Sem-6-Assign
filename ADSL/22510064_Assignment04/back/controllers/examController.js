// controllers/examController.js
const db = require('../db');

exports.createExam = (req, res) => {
  const { course_id, created_by, title, description, total_marks, start_time, duration_minutes, question_ids, assigned_students } = req.body;
  
  // Validate required fields
  if (!course_id || !created_by || !title || !question_ids || !assigned_students) {
    return res.status(400).send({ error: 'Missing required fields' });
  }

  const sql = 'INSERT INTO exams (course_id, created_by, title, description, total_marks, start_time, duration_minutes, question_ids, assigned_students, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)';
  db.query(sql, [
    course_id,
    created_by,
    title,
    description,
    total_marks,
    start_time,
    duration_minutes,
    JSON.stringify(question_ids),
    JSON.stringify(assigned_students),
    'scheduled' // Default status
  ], (err, result) => {
    if (err) return res.status(500).send(err);
    res.send({ message: 'Exam created successfully', examId: result.insertId });
  });
};


exports.getExams = (req, res) => {
  const sql = 'SELECT * FROM exams';
  db.query(sql, (err, results) => {
    if (err) return res.status(500).send(err);
    results.forEach(exam => {
      if (exam.question_ids) exam.question_ids = JSON.parse(exam.question_ids);
      if (exam.assigned_students) exam.assigned_students = JSON.parse(exam.assigned_students);
    });
    res.send(results);
  });
};

exports.getExamById = (req, res) => {
  const examId = req.params.id;
  const sql = `SELECT e.*, 
                (SELECT COUNT(*) FROM student_exam_attempts 
                 WHERE exam_id = e.id) AS attempts_count,
                (SELECT AVG(percentage_score) FROM student_exam_attempts 
                 WHERE exam_id = e.id) AS average_score
               FROM exams e WHERE id = ?`;
  db.query(sql, [examId], (err, results) => {
    if (err) return res.status(500).send(err);
    if (results.length === 0) return res.status(404).send({ error: 'Exam not found' });
    const exam = results[0];
    if (exam.question_ids) exam.question_ids = JSON.parse(exam.question_ids);
    if (exam.assigned_students) exam.assigned_students = JSON.parse(exam.assigned_students);
    res.send(exam);
  });
};

exports.createExam = (req, res) => {
  const { course_id, created_by, title, description, total_marks, start_time, duration_minutes, question_ids, assigned_students } = req.body;
  
  // Validate required fields
  if (!course_id || !created_by || !title || !question_ids || !assigned_students) {
    return res.status(400).send({ error: 'Missing required fields' });
  }

  const sql = 'INSERT INTO exams (course_id, created_by, title, description, total_marks, start_time, duration_minutes, question_ids, assigned_students, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)';
  db.query(sql, [
    course_id,
    created_by,
    title,
    description,
    total_marks,
    start_time,
    duration_minutes,
    JSON.stringify(question_ids),
    JSON.stringify(assigned_students),
    'scheduled' // Default status
  ], (err, result) => {
    if (err) return res.status(500).send(err);
    res.send({ message: 'Exam created successfully', examId: result.insertId });
  });
};


exports.getExams = (req, res) => {
  const sql = 'SELECT * FROM exams';
  db.query(sql, (err, results) => {
    if (err) return res.status(500).send(err);
    results.forEach(exam => {
      if (exam.question_ids) exam.question_ids = JSON.parse(exam.question_ids);
      if (exam.assigned_students) exam.assigned_students = JSON.parse(exam.assigned_students);
    });
    res.send(results);
  });
};

exports.getExamById = (req, res) => {
  const examId = req.params.id;
  const sql = `SELECT e.*, 
                (SELECT COUNT(*) FROM student_exam_attempts 
                 WHERE exam_id = e.id) AS attempts_count,
                (SELECT AVG(percentage_score) FROM student_exam_attempts 
                 WHERE exam_id = e.id) AS average_score
               FROM exams e WHERE id = ?`;
  db.query(sql, [examId], (err, results) => {
    if (err) return res.status(500).send(err);
    if (results.length === 0) return res.status(404).send({ error: 'Exam not found' });
    const exam = results[0];
    if (exam.question_ids) exam.question_ids = JSON.parse(exam.question_ids);
    if (exam.assigned_students) exam.assigned_students = JSON.parse(exam.assigned_students);
    res.send(exam);
  });
};

// New method to update exam details
exports.updateExam = (req, res) => {
  const examId = req.params.id;
  const { title, description, total_marks, start_time, duration_minutes, question_ids, assigned_students } = req.body;
  
  const sql = `UPDATE exams SET 
                title = ?, 
                description = ?, 
                total_marks = ?, 
                start_time = ?, 
                duration_minutes = ?, 
                question_ids = ?, 
                assigned_students = ?
              WHERE id = ?`;
  db.query(sql, [
    title,
    description,
    total_marks,
    start_time,
    duration_minutes,
    JSON.stringify(question_ids),
    JSON.stringify(assigned_students),
    examId
  ], (err, result) => {
    if (err) return res.status(500).send(err);
    if (result.affectedRows === 0) return res.status(404).send({ error: 'Exam not found' });
    res.send({ message: 'Exam updated successfully' });
  });
};


exports.submitExam = (req, res) => {
  const { exam_id, student_id, answers } = req.body;
  
  // Validate required fields
  if (!exam_id || !student_id || !answers) {
    return res.status(400).send({ error: 'Missing required fields' });
  }

  // Get exam details
  const getExamSql = 'SELECT * FROM exams WHERE id = ?';
  db.query(getExamSql, [exam_id], (err, examResults) => {
    if (err) return res.status(500).send(err);
    if (examResults.length === 0) return res.status(404).send({ error: 'Exam not found' });
    
    const exam = examResults[0];
    const questionIds = JSON.parse(exam.question_ids);
    
    // Calculate score based on answers
    let score = 0;
    const correctAnswers = {}; // This should be fetched from questions table
    for (const [questionId, answer] of Object.entries(answers)) {
      if (correctAnswers[questionId] === answer) {
        score += 1; // Assuming each question is worth 1 point
      }
    }

    // Calculate percentage score
    const totalQuestions = questionIds.length;
    const percentageScore = (score / totalQuestions) * 100;

    // Update exam attempt
    const sql = `INSERT INTO student_exam_attempts 
                (student_id, exam_id, status, score, total_questions, correct_answers, percentage_score) 
                VALUES (?, ?, ?, ?, ?, ?, ?)`;
    db.query(sql, [
      student_id,
      exam_id,
      'completed',
      score,
      totalQuestions,
      score,
      percentageScore
    ], (err, result) => {
      if (err) return res.status(500).send(err);
      
      // Update exam status if all students have completed
      const updateExamSql = `UPDATE exams SET status = 'completed' 
                           WHERE id = ? AND NOT EXISTS (
                             SELECT 1 FROM student_exam_attempts 
                             WHERE exam_id = ? AND status != 'completed'
                           )`;
      db.query(updateExamSql, [exam_id, exam_id], (err, updateResult) => {
        if (err) return res.status(500).send(err);
        res.send({ 
          message: 'Exam submitted successfully', 
          score,
          totalQuestions,
          percentageScore
        });
      });
    });
  });
};
