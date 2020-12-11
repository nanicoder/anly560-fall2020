package ooExamples;

public class Car {
	
	//Fields for the class / attributes
	private String name;
	private double topSpeed;
	
	// constructor for the class Car
	public Car() {
		
	}
	
	public String getName() {
		return name;
	}
	
	public void setName(String newName) {
		this.name = newName;
	}
	
	public void setTopSpeed(double speedMPH) {
		topSpeed = speedMPH;
	}
	public double getTopSpeed() {
		return topSpeed;
	
	}
	public double getTopSpeedMPH() {
		return topSpeed;
	
	}
	public double getTopSpeedKPH() {
		return topSpeed * 1.609344;
	}
	

}
