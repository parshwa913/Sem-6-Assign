// controllers/questionController.js
const db = require('../db');

exports.addQuestion = (req, res) => {
  const { course_id, created_by, question_text, option_a, option_b, option_c, option_d, correct_option, difficulty } = req.body;
  const image_url = req.file ? req.file.filename : null;
  const sql = 'INSERT INTO questions (course_id, created_by, question_text, option_a, option_b, option_c, option_d, correct_option, image_url, difficulty) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)';
  db.query(sql, [course_id, created_by, question_text, option_a, option_b, option_c, option_d, correct_option, image_url, difficulty], (err, result) => {
    if (err) return res.status(500).send(err);
    res.send({ message: 'Question added successfully', questionId: result.insertId });
  });
};

exports.getQuestions = (req, res) => {
  const sql = 'SELECT * FROM questions';
  db.query(sql, (err, results) => {
    if (err) return res.status(500).send(err);
    res.send(results);
  });
};

exports.updateQuestion = (req, res) => {
  const questionId = req.params.id;
  const { question_text, option_a, option_b, option_c, option_d, correct_option, difficulty } = req.body;
  const image_url = req.file ? req.file.filename : null;
  const sql = 'UPDATE questions SET question_text = ?, option_a = ?, option_b = ?, option_c = ?, option_d = ?, correct_option = ?, difficulty = ?, image_url = ? WHERE id = ?';
  db.query(sql, [question_text, option_a, option_b, option_c, option_d, correct_option, difficulty, image_url, questionId], (err, result) => {
    if (err) return res.status(500).send(err);
    res.send({ message: 'Question updated successfully' });
  });
};
