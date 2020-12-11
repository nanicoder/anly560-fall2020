-- Test db to practice connecting to mysql with python

DROP SCHEMA IF EXISTS marathondb;
CREATE SCHEMA marathondb;
USE marathondb;

--
-- Table structure for table person
--

CREATE TABLE person (
  person_id  VARCHAR(20) not null,
  first_name VARCHAR(20) ,
  last_name VARCHAR(20),
  sex char(1) 
);


--
-- Table structure for table marathon
--
CREATE TABLE marathon (
  person_id  varchar(20) not null,
  raceYear VARCHAR(4) not null,
  raceTime	float 
);

-- -------------------
-- Define primary keys
-- -------------------
alter table person add primary key (person_id);
alter table marathon add primary key (person_id, raceYear);

-- -------------------
-- Define foreign keys
-- -------------------
alter table marathon add constraint FL_marathon_person foreign key (person_id) references person (person_id); 

##create a user
DROP USER IF EXISTS 'marathonuser'@'localhost';
CREATE USER 'marathonuser'@'localhost' IDENTIFIED BY 'test123';
GRANT ALL PRIVILEGES ON marathondb.* TO 'marathonuser'@'localhost';