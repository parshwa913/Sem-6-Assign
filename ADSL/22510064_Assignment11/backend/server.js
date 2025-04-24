const express = require('express');
const neo4j = require('neo4j-driver');
const cors = require('cors');
require('dotenv').config();

const app = express();

app.use(cors());
app.use(express.json());

const driver = neo4j.driver(
    process.env.NEO4J_URI,
    neo4j.auth.basic(process.env.NEO4J_USER, process.env.NEO4J_PASSWORD)
);

const testConnection = async () => {
    const session = driver.session();
    try {
        await session.run('RETURN 1');
        console.log('Neo4j connection successful');
    } catch (error) {
        console.error('Neo4j connection error:', error);
    } finally {
        await session.close();
    }
};

testConnection();

app.post('/api/check-citation', async (req, res) => {
    const session = driver.session();
    const { paperA, paperB } = req.body;

    try {
        const result = await session.run(
            `MATCH (a:Paper {paper_id: $paperA}), (b:Paper {paper_id: $paperB})
             OPTIONAL MATCH (author_a:Author)-[:WROTE]->(a)
             OPTIONAL MATCH (author_b:Author)-[:WROTE]->(b)
             OPTIONAL MATCH (a)-[cite:CITES]->(b)
             OPTIONAL MATCH (b)-[cite2:CITES]->(a)
             RETURN a, b, author_a, author_b, cite, cite2`,
            { paperA, paperB }
        );

        if (result.records.length > 0 && 
            (result.records[0].get('cite') || result.records[0].get('cite2'))) {
            res.send('yes');
        } else {
            res.send('no');
        }
    } catch (error) {
        console.error('Query error:', error);
        res.status(500).json({ error: 'Error executing query' });
    } finally {
        await session.close();
    }
});

// Get paper classification
app.post('/api/get-classification', async (req, res) => {
    const session = driver.session();
    const { paperId } = req.body;

    try {
        const result = await session.run(
            `MATCH (p:Paper {paper_id: $paperId})
             RETURN p.paper_id, p.title, p.filename, p.classification`,
            { paperId }
        );

        if (result.records.length > 0) {
            const record = result.records[0];
            const paperInfo = {
                paper_id: record.get('p.paper_id'),
                title: record.get('p.title'),
                filename: record.get('p.filename'),
                classification: record.get('p.classification')
            };
            res.json(paperInfo);
        } else {
            res.json({
                message: 'No classification found for this paper'
            });
        }
    } catch (error) {
        console.error('Query error:', error);
        res.status(500).json({ error: 'Error executing query' });
    } finally {
        await session.close();
    }
});

// Get all papers endpoint
app.get('/api/all-papers', async (req, res) => {
    const session = driver.session();

    // Fix: Ensure limit and skip are valid integers
    let limit = 50;
    let skip = 0;

    try {
        if (req.query.limit) {
            const parsedLimit = Math.floor(Number(req.query.limit));
            if (!isNaN(parsedLimit) && parsedLimit >= 0) {
                limit = parsedLimit;
            }
        }

        if (req.query.skip) {
            const parsedSkip = Math.floor(Number(req.query.skip));
            if (!isNaN(parsedSkip) && parsedSkip >= 0) {
                skip = parsedSkip;
            }
        }
    } catch (e) {
        console.error('Error parsing pagination parameters:', e);
    }

    try {
        const result = await session.run(
            `MATCH (p:Paper)
             WHERE p.title IS NOT NULL
             RETURN p.paper_id, p.title, p.filename, p.classification
             ORDER BY p.paper_id
             SKIP $skip
             LIMIT $limit`,
            { skip: neo4j.int(skip), limit: neo4j.int(limit) }
        );

        const papers = result.records.map(record => ({
            paper_id: record.get('p.paper_id'),
            title: record.get('p.title'),
            filename: record.get('p.filename'),
            classification: record.get('p.classification')
        }));

        res.json(papers);
    } catch (error) {
        console.error('Query error:', error);
        res.status(500).json({ error: 'Error executing query' });
    } finally {
        await session.close();
    }
});

app.post('/api/paper-details', async (req, res) => {
    const session = driver.session();
    const { paperId } = req.body;

    try {
        const result = await session.run(
            `MATCH (p:Paper {paper_id: $paperId})
             OPTIONAL MATCH (a:Author)-[:WROTE]->(p)
             OPTIONAL MATCH (p)-[:CITES]->(cited:Paper)
             OPTIONAL MATCH (citing:Paper)-[:CITES]->(p)
             RETURN p,
                    collect(DISTINCT a) as authors,
                    collect(DISTINCT cited) as citedPapers,
                    collect(DISTINCT citing) as citingPapers`,
            { paperId }
        );

        if (result.records.length > 0) {
            const record = result.records[0];
            const paper = record.get('p').properties;
            const authors = record.get('authors').map(a => a.properties);
            const citedPapers = record.get('citedPapers').map(p => p.properties);
            const citingPapers = record.get('citingPapers').map(p => p.properties);

            res.json({
                paper: paper,
                authors: authors,
                citedPapers: citedPapers,
                citingPapers: citingPapers
            });
        } else {
            res.json({ message: 'Paper not found' });
        }
    } catch (error) {
        console.error('Query error:', error);
        res.status(500).json({ error: 'Error executing query' });
    } finally {
        await session.close();
    }
});


// Start the server
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});
